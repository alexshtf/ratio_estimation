"""Online ratio models built around proximal updates."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .proximal import (
    exponential_prox,
    positive_square_prox,
    prox_affine_composition,
    prox_with_regularizer_shift,
    softplus_integral_prox,
    square_prox,
)

FloatArray = NDArray[np.float64]


def serialize_array(values: FloatArray | None) -> list[float] | None:
    """Convert a NumPy array into a JSON-friendly list."""
    return None if values is None else values.tolist()


class PositiveLink(Protocol):
    """A positive link function together with its scalar proximal operator."""

    def prox(self, x: ArrayLike, step_size: float) -> FloatArray:
        """Apply the prox of the integrated link."""
        ...

    def predict(self, x: ArrayLike) -> FloatArray:
        """Map latent values to positive ratios."""
        ...


class ExponentialLink:
    """Predict ratios through exp(z)."""

    def prox(self, x: ArrayLike, step_size: float) -> FloatArray:
        """Apply the prox of the exponential integral."""
        return exponential_prox(x, step_size)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Map latent values to positive ratios."""
        return np.exp(np.asarray(x, dtype=float))


class PositivePartLink:
    """Predict ratios through max(0, z)."""

    def prox(self, x: ArrayLike, step_size: float) -> FloatArray:
        """Apply the prox of the positive square."""
        return positive_square_prox(x, step_size)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Map latent values to positive ratios."""
        return np.maximum(0.0, np.asarray(x, dtype=float))


class SoftplusLink:
    """Predict ratios through log(1 + exp(z))."""

    def prox(self, x: ArrayLike, step_size: float) -> FloatArray:
        """Apply the prox of the softplus integral."""
        return softplus_integral_prox(x, step_size)

    def predict(self, x: ArrayLike) -> FloatArray:
        """Map latent values to positive ratios."""
        return np.logaddexp(0.0, np.asarray(x, dtype=float))


@dataclass(slots=True)
class RatioProximalLearner:
    """Fit a direct ratio model r̂(x) = f′(<w, x>) with proximal updates."""

    link: PositiveLink
    step_size: float
    regularization: float
    weights: FloatArray | None = field(default=None, init=False)

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)

        def scalar_loss_prox(z: ArrayLike, prox_step_size: float) -> FloatArray:
            return prox_affine_composition(
                z,
                prox_step_size * denominator,
                self.link.prox,
                features,
            )

        self.weights = prox_with_regularizer_shift(
            self.weights,
            self.step_size,
            scalar_loss_prox,
            shift=-numerator * features,
            regularization=self.regularization,
        )

    def predict(self, x: ArrayLike) -> float:
        """Predict the ratio for one feature vector."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)
        score = float(np.dot(self.weights, features))
        return float(self.link.predict(score))

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current learner state."""
        return {
            "link": type(self.link).__name__,
            "step_size": self.step_size,
            "regularization": self.regularization,
            "weights": serialize_array(self.weights),
        }


@dataclass(slots=True)
class LinearRatioLearner:
    """Fit a linear ratio model by squared loss on denominator · ratio."""

    step_size: float
    regularization: float
    weights: FloatArray | None = field(default=None, init=False)

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)

        def scalar_loss_prox(z: ArrayLike, prox_step_size: float) -> FloatArray:
            return prox_affine_composition(z, prox_step_size, square_prox, denominator * features)

        self.weights = prox_with_regularizer_shift(
            self.weights,
            self.step_size,
            scalar_loss_prox,
            shift=-numerator * denominator * features,
            regularization=self.regularization,
        )

    def predict(self, x: ArrayLike) -> float:
        """Predict the ratio for one feature vector."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)
        return float(np.dot(self.weights, features))

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current learner state."""
        return {
            "step_size": self.step_size,
            "regularization": self.regularization,
            "weights": serialize_array(self.weights),
        }


@dataclass(slots=True)
class LinearInverseRatioLearner:
    """Fit a linear inverse-ratio model by squared loss on numerator / ratio."""

    step_size: float
    regularization: float
    weights: FloatArray | None = field(default=None, init=False)

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Fit one streaming observation."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)

        def scalar_loss_prox(z: ArrayLike, prox_step_size: float) -> FloatArray:
            return prox_affine_composition(z, prox_step_size, square_prox, numerator * features)

        self.weights = prox_with_regularizer_shift(
            self.weights,
            self.step_size,
            scalar_loss_prox,
            shift=-numerator * denominator * features,
            regularization=self.regularization,
        )

    def predict(self, x: ArrayLike) -> float:
        """Predict the ratio for one feature vector."""
        features = np.asarray(x, dtype=float)
        if self.weights is None:
            self.weights = np.zeros_like(features)
        return float(np.reciprocal(np.dot(self.weights, features)))

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the current learner state."""
        return {
            "step_size": self.step_size,
            "regularization": self.regularization,
            "weights": serialize_array(self.weights),
        }
