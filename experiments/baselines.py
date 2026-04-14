"""Baseline models used for experiment comparisons."""

import sys
from enum import Enum
from typing import cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.special import wrightomega
from sklearn.linear_model import SGDRegressor

from ratio_estimation._state import state_snapshot

MAX_LOG_FLOAT = float(np.log(sys.float_info.max))


class RatioOfRegressorsBaseline:
    """Fit numerator and denominator models separately and divide them."""

    def __init__(
        self,
        dimension: int,
        numerator_step_size: float,
        numerator_regularization: float,
        denominator_step_size: float,
        denominator_regularization: float,
        epsilon: float = 1e-10,
    ) -> None:
        self.dimension = dimension
        self.numerator_regressor = SGDRegressor(
            loss="squared_error",
            alpha=numerator_regularization,
            learning_rate="constant",
            eta0=numerator_step_size,
        )
        self.denominator_regressor = SGDRegressor(
            loss="squared_error",
            alpha=denominator_regularization,
            learning_rate="constant",
            eta0=denominator_step_size,
        )
        self.epsilon = epsilon
        self.is_fitted = False

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float).reshape(1, -1)
        log_numerator = np.array([np.log(self.epsilon + numerator)], dtype=float)
        log_denominator = np.array([np.log(self.epsilon + denominator)], dtype=float)
        self.numerator_regressor.partial_fit(features, log_numerator)
        self.denominator_regressor.partial_fit(features, log_denominator)
        self.is_fitted = True

    def predict(self, x: ArrayLike) -> float:
        """Predict the ratio from separate log-scale regressors."""
        if not self.is_fitted:
            return 1.0

        features = np.asarray(x, dtype=float).reshape(-1)
        log_numerator = self._linear_predict(self.numerator_regressor, features)
        log_denominator = self._linear_predict(self.denominator_regressor, features)
        return float(np.exp(min(log_numerator - log_denominator, MAX_LOG_FLOAT)))

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        numerator_coef = (
            self.numerator_regressor.coef_
            if self.is_fitted
            else np.zeros(self.dimension, dtype=float)
        )
        numerator_intercept = (
            self.numerator_regressor.intercept_
            if self.is_fitted
            else np.zeros(1, dtype=float)
        )
        denominator_coef = (
            self.denominator_regressor.coef_
            if self.is_fitted
            else np.zeros(self.dimension, dtype=float)
        )
        denominator_intercept = (
            self.denominator_regressor.intercept_
            if self.is_fitted
            else np.zeros(1, dtype=float)
        )
        return state_snapshot(
            epsilon=self.epsilon,
            numerator_coef=numerator_coef,
            numerator_intercept=numerator_intercept,
            denominator_coef=denominator_coef,
            denominator_intercept=denominator_intercept,
        )

    @staticmethod
    def _linear_predict(regressor: SGDRegressor, features: np.ndarray) -> float:
        """Return the raw linear score without sklearn's prediction wrapper."""
        coefficients = cast(np.ndarray, regressor.coef_)
        intercept = cast(np.ndarray, regressor.intercept_)
        return float(np.dot(coefficients, features) + intercept[0])


class CampaignRunningRatioBaseline:
    """Predict each campaign's cumulative observed ratio before the current update."""

    def __init__(self, default_prediction: float = 1.0) -> None:
        self.default_prediction = default_prediction
        self.cumulative_numerator = 0.0
        self.cumulative_denominator = 0.0

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Accumulate one streaming observation."""
        _ = x
        self.cumulative_numerator += numerator
        self.cumulative_denominator += denominator

    def predict(self, x: ArrayLike | None = None) -> float:
        """Return the cumulative ratio observed before the current update."""
        _ = x
        if self.cumulative_denominator <= 0.0:
            return self.default_prediction
        return float(self.cumulative_numerator / self.cumulative_denominator)

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            default_prediction=self.default_prediction,
            cumulative_numerator=self.cumulative_numerator,
            cumulative_denominator=self.cumulative_denominator,
        )


class QuadraticRatioBaseline:
    """Fit a positive linear ratio model with a quadratic loss."""

    def __init__(self, dimension: int, step_size: float, regularization: float) -> None:
        self.weights = np.zeros(dimension, dtype=float)
        self.bias = 1.0
        self.step_size = step_size
        self.regularization = regularization
        self.iteration = 1

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        step = self.step_size / np.sqrt(self.iteration)
        scale = (
            -numerator
            if denominator == 0
            else self._dual_scale(features, numerator, denominator, step)
        )
        shrink = 1.0 + step * self.regularization
        self.weights = (self.weights - step * scale * features) / shrink
        self.bias = (self.bias - step * scale) / shrink
        self.iteration += 1

    def predict(self, x: ArrayLike) -> float:
        """Predict a nonnegative ratio."""
        features = np.asarray(x, dtype=float)
        score = float(np.dot(self.weights, features) + self.bias)
        return max(0.0, score)

    def _dual_scale(
        self,
        x: ArrayLike,
        numerator: float,
        denominator: float,
        step: float,
    ) -> float:
        score = float(np.dot(self.weights, x) + self.bias)
        feature_norm = 1.0 + float(np.dot(x, x))
        first_candidate = denominator * score - numerator - step * numerator * self.regularization
        first_candidate /= denominator * step * feature_norm + 1.0 + step * self.regularization
        second_candidate = score / (step * feature_norm)
        if first_candidate >= -numerator:
            return float(first_candidate)
        if second_candidate < -numerator:
            return float(second_candidate)
        return float(-numerator)

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            weights=self.weights,
            bias=self.bias,
            step_size=self.step_size,
            regularization=self.regularization,
            iteration=self.iteration,
        )


class ExponentialRatioBaseline:
    """Fit an exponential ratio model with closed-form dual updates."""

    def __init__(self, dimension: int, step_size: float, regularization: float) -> None:
        self.weights = np.zeros(dimension, dtype=float)
        self.bias = 1.0
        self.step_size = step_size
        self.regularization = regularization
        self.iteration = 1

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        step = self.step_size / np.sqrt(self.iteration)
        shrink = 1.0 + step * self.regularization
        dual = self._dual_solution(features, numerator, denominator, step, shrink)
        self.weights = (self.weights - step * dual * features) / shrink
        self.bias = (self.bias - step * dual) / shrink
        self.iteration += 1

    def predict(self, x: ArrayLike) -> float:
        """Predict a positive ratio."""
        features = np.asarray(x, dtype=float)
        score = float(np.dot(self.weights, features) + self.bias)
        return float(np.exp(min(score, MAX_LOG_FLOAT)))

    def _dual_solution(
        self,
        x: ArrayLike,
        numerator: float,
        denominator: float,
        step: float,
        shrink: float,
    ) -> float:
        if denominator == 0:
            return float(-numerator)

        quadratic_scale = step * (1.0 + float(np.dot(x, x))) / shrink
        score = float(np.dot(self.weights, x) + self.bias) / shrink
        return float(
            wrightomega(numerator * quadratic_scale + score + np.log(denominator * quadratic_scale))
            / quadratic_scale
            - numerator
        )

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            weights=self.weights,
            bias=self.bias,
            step_size=self.step_size,
            regularization=self.regularization,
            iteration=self.iteration,
        )


class ExponentialQuadraticBaseline:
    """Fit a hybrid link that is exponential on the left and quadratic on the right."""

    def __init__(self, dimension: int, step_size: float, regularization: float) -> None:
        self.weights = np.zeros(dimension, dtype=float)
        self.bias = 1.0
        self.step_size = step_size
        self.regularization = regularization
        self.iteration = 1

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        step = self.step_size / np.sqrt(self.iteration)
        shrink = 1.0 + step * self.regularization
        dual = self._dual_solution(features, numerator, denominator, step, shrink)
        self.weights = (self.weights - step * dual * features) / shrink
        self.bias = (self.bias - step * dual) / shrink
        self.iteration += 1

    def predict(self, x: ArrayLike) -> float:
        """Predict a positive ratio with the hybrid link."""
        features = np.asarray(x, dtype=float)
        score = float(np.dot(self.weights, features) + self.bias)
        return float(1.0 + score + 0.5 * score**2) if score > 0 else float(np.exp(score))

    def _dual_solution(
        self,
        x: ArrayLike,
        numerator: float,
        denominator: float,
        step: float,
        shrink: float,
    ) -> float:
        if denominator == 0:
            return float(-numerator)

        quadratic_scale = step * (1.0 + float(np.dot(x, x))) / shrink
        score = float(np.dot(self.weights, x) + self.bias) / shrink

        def conjugate_derivative(s: float) -> float:
            return float(np.log(s)) if 0.0 < s <= 1.0 else float(s - 1.0)

        def derivative(dual: float) -> float:
            return (
                quadratic_scale * dual
                - score
                + conjugate_derivative((dual + numerator) / denominator)
            )

        lower_bound = 1.0
        while derivative(-numerator + lower_bound) >= 0.0:
            lower_bound /= 2.0
        lower_bound = -numerator + lower_bound

        upper_bound = 1.0
        while derivative(upper_bound) <= 0.0:
            upper_bound *= 2.0

        root, _ = brentq(derivative, lower_bound, upper_bound, full_output=True)
        return float(root)

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            weights=self.weights,
            bias=self.bias,
            step_size=self.step_size,
            regularization=self.regularization,
            iteration=self.iteration,
        )


class LinearRegressionBaseline:
    """Fit either a linear ratio or a linear inverse-ratio baseline."""

    def __init__(
        self,
        dimension: int,
        step_size: float,
        regularization: float,
        inverse: bool,
    ) -> None:
        self.weights = np.zeros(dimension, dtype=float)
        self.bias = 1.0
        self.step_size = step_size
        self.regularization = regularization
        self.inverse = inverse
        self.iteration = 1

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        alpha, beta = (numerator, denominator) if self.inverse else (denominator, numerator)
        features = np.asarray(x, dtype=float)
        step = self.step_size / np.sqrt(self.iteration)
        shrink = 1.0 + step * self.regularization
        dual = 0.0 if alpha == 0 else self._dual_solution(features, alpha, beta, step, shrink)
        self.weights = (self.weights - step * dual * features) / shrink
        self.bias = (self.bias - step * dual) / shrink
        self.iteration += 1

    def predict(self, x: ArrayLike) -> float:
        """Predict the ratio or inverse-ratio baseline."""
        features = np.asarray(x, dtype=float)
        score = float(np.dot(self.weights, features) + self.bias)
        if self.inverse:
            return float(1.0 / score) if score > 0 else 0.0
        return float(score) if score > 0 else 0.0

    def _dual_solution(
        self,
        x: ArrayLike,
        alpha: float,
        beta: float,
        step: float,
        shrink: float,
    ) -> float:
        quadratic_scale = step * (1.0 + float(np.dot(x, x))) / shrink
        score = float(np.dot(self.weights, x) + self.bias) / shrink
        alpha_beta = alpha * beta
        alpha_squared = alpha**2

        first_candidate = (alpha_squared * score - alpha_beta) / (
            alpha_squared * quadratic_scale + 1.0
        )
        second_candidate = score / quadratic_scale
        return float(first_candidate) if first_candidate >= -alpha_beta else float(second_candidate)

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            weights=self.weights,
            bias=self.bias,
            step_size=self.step_size,
            regularization=self.regularization,
            inverse=self.inverse,
            iteration=self.iteration,
        )


class DecayMode(Enum):
    """Choose which quantity triggers ratio decay."""

    COST = "cost"
    COUNT = "count"
    TIME = "time"


class DecayRatioBaseline:
    """Estimate a ratio with periodic multiplicative decay."""

    def __init__(self, decay_rate: float, decay_interval: float, mode: DecayMode) -> None:
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.mode = mode
        self.numerator = 0.0
        self.denominator = 0.0
        self.current_interval: float | None = None

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Accumulate one observation and decay when the interval is crossed."""
        time = float(np.asarray(x, dtype=float).reshape(-1)[0])

        if self.mode is DecayMode.COST:
            self.current_interval = 0.0 if self.current_interval is None else self.current_interval
            self.current_interval += numerator
            self.numerator += numerator
            self.denominator += denominator
            self._apply_interval_decays()
        elif self.mode is DecayMode.COUNT:
            self.current_interval = 0.0 if self.current_interval is None else self.current_interval
            self.current_interval += denominator
            self.numerator += numerator
            self.denominator += denominator
            self._apply_interval_decays()
        else:
            if self.current_interval is None:
                self.current_interval = time
            else:
                elapsed_time = time - self.current_interval
                decay_steps = int(elapsed_time // self.decay_interval)
                if decay_steps > 0:
                    self._decay(decay_steps)
                    self.current_interval += decay_steps * self.decay_interval
            self.numerator += numerator
            self.denominator += denominator

    def predict(self, x: ArrayLike | None = None) -> float:
        """Return the current decayed ratio estimate."""
        _ = x
        return 0.0 if self.denominator == 0 else float(self.numerator / self.denominator)

    def _apply_interval_decays(self) -> None:
        """Apply one decay per fully elapsed accumulated interval."""
        assert self.current_interval is not None
        decay_steps = int(self.current_interval // self.decay_interval)
        if decay_steps <= 0:
            return
        self._decay(decay_steps)
        self.current_interval -= decay_steps * self.decay_interval

    def _decay(self, steps: int = 1) -> None:
        """Decay the running numerator and denominator by several intervals."""
        decay_scale = self.decay_rate**steps
        self.numerator *= decay_scale
        self.denominator *= decay_scale

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current baseline state."""
        return state_snapshot(
            decay_rate=self.decay_rate,
            decay_interval=self.decay_interval,
            mode=self.mode,
            numerator=self.numerator,
            denominator=self.denominator,
            current_interval=self.current_interval,
        )
