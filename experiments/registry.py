"""Shared model registry for the maintained experiment workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import optuna
from numpy.typing import ArrayLike

from ratio_estimation.models import LinearRatioLearner, RatioProximalLearner, SoftplusLink

from .baselines import (
    DecayMode,
    DecayRatioBaseline,
    ExponentialQuadraticBaseline,
    ExponentialRatioBaseline,
    LinearRegressionBaseline,
    QuadraticRatioBaseline,
    RatioOfRegressorsBaseline,
)

Params = dict[str, Any]
SuggestParams = Callable[[optuna.Trial], Params]


class StreamingModelLike(Protocol):
    """A model instance returned by the experiment registry."""

    def predict(self, x: ArrayLike) -> float:
        """Predict one ratio."""
        ...

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Update the model with one observation."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the model state."""
        ...


BuildModel = Callable[[Params], StreamingModelLike]

BENCHMARK_MODEL_NAMES = (
    "exponential",
    "inverse_exponential",
    "ratio_of_regressors",
    "quadratic",
    "linear",
    "inverse_linear",
    "decay_cost",
    "decay_count",
    "decay_time",
)

SINGLE_STREAM_EXTRA_MODEL_NAMES = ("exponential_quadratic",)


@dataclass(frozen=True, slots=True)
class ExperimentModelSpec:
    """One tuned model family in the maintained experiment suite."""

    name: str
    input_column: str
    suggest_params: SuggestParams
    build_model: BuildModel


def proximal_softplus_spec() -> ExperimentModelSpec:
    """Return the maintained spec for the softplus proximal learner."""
    return ExperimentModelSpec(
        name="proximal_softplus",
        input_column="features",
        suggest_params=_suggest_step_regularized,
        build_model=lambda params: RatioProximalLearner(
            link=SoftplusLink(),
            step_size=params["step_size"],
            regularization=params["regularization"],
        ),
    )


def linear_ratio_spec() -> ExperimentModelSpec:
    """Return the maintained spec for the direct linear ratio learner."""
    return ExperimentModelSpec(
        name="linear_ratio",
        input_column="features",
        suggest_params=_suggest_step_regularized,
        build_model=lambda params: LinearRatioLearner(
            step_size=params["step_size"],
            regularization=params["regularization"],
        ),
    )


def _suggest_step_regularized(trial: optuna.Trial) -> Params:
    """Suggest the common step-size and regularization hyperparameters."""
    return {
        "step_size": trial.suggest_float("step_size", 1e-8, 100.0, log=True),
        "regularization": trial.suggest_float("regularization", 1e-10, 100.0, log=True),
    }


def _suggest_decay_params(trial: optuna.Trial) -> Params:
    """Suggest the common hyperparameters for decay baselines."""
    return {
        "decay_rate": trial.suggest_float("decay_rate", 0.0, 1.0),
        "decay_interval": trial.suggest_float("decay_interval", 1.0, 1000.0, log=True),
    }


def _suggest_exponential_quadratic_params(trial: optuna.Trial) -> Params:
    """Suggest the maintained single-stream search space for the hybrid baseline."""
    return {
        "step_size": trial.suggest_float("step_size", 1e-4, 100.0, log=True),
        "regularization": trial.suggest_float("regularization", 1e-10, 0.1, log=True),
    }


def _step_regularized_spec(
    name: str,
    history_length: int,
    build_model: Callable[[int, Params], StreamingModelLike],
) -> ExperimentModelSpec:
    """Build a spec for models with the common step-size/regularization search space."""
    return ExperimentModelSpec(
        name=name,
        input_column="features",
        suggest_params=_suggest_step_regularized,
        build_model=lambda params: build_model(history_length, params),
    )


def _decay_spec(name: str, mode: DecayMode) -> ExperimentModelSpec:
    """Build a spec for one decay baseline."""
    return ExperimentModelSpec(
        name=name,
        input_column="offset",
        suggest_params=_suggest_decay_params,
        build_model=lambda params: DecayRatioBaseline(
            decay_rate=params["decay_rate"],
            decay_interval=params["decay_interval"],
            mode=mode,
        ),
    )


def build_model_registry(history_length: int) -> dict[str, ExperimentModelSpec]:
    """Build the shared registry used by the maintained experiment entrypoints."""
    specs = [
        _step_regularized_spec(
            "exponential",
            history_length,
            lambda dimension, params: ExponentialRatioBaseline(
                dimension=dimension,
                step_size=params["step_size"],
                regularization=params["regularization"],
                inverse=False,
            ),
        ),
        _step_regularized_spec(
            "inverse_exponential",
            history_length,
            lambda dimension, params: ExponentialRatioBaseline(
                dimension=dimension,
                step_size=params["step_size"],
                regularization=params["regularization"],
                inverse=True,
            ),
        ),
        ExperimentModelSpec(
            name="ratio_of_regressors",
            input_column="features",
            suggest_params=lambda trial: {
                "numerator_step_size": trial.suggest_float(
                    "numerator_step_size",
                    1e-8,
                    100.0,
                    log=True,
                ),
                "numerator_regularization": trial.suggest_float(
                    "numerator_regularization",
                    1e-10,
                    100.0,
                    log=True,
                ),
                "denominator_step_size": trial.suggest_float(
                    "denominator_step_size",
                    1e-8,
                    100.0,
                    log=True,
                ),
                "denominator_regularization": trial.suggest_float(
                    "denominator_regularization",
                    1e-10,
                    100.0,
                    log=True,
                ),
                "epsilon": trial.suggest_float("epsilon", 1e-10, 1.0, log=True),
            },
            build_model=lambda params: RatioOfRegressorsBaseline(
                dimension=history_length,
                numerator_step_size=params["numerator_step_size"],
                numerator_regularization=params["numerator_regularization"],
                denominator_step_size=params["denominator_step_size"],
                denominator_regularization=params["denominator_regularization"],
                epsilon=params["epsilon"],
            ),
        ),
        _step_regularized_spec(
            "quadratic",
            history_length,
            lambda dimension, params: QuadraticRatioBaseline(
                dimension=dimension,
                step_size=params["step_size"],
                regularization=params["regularization"],
            ),
        ),
        _step_regularized_spec(
            "linear",
            history_length,
            lambda dimension, params: LinearRegressionBaseline(
                dimension=dimension,
                step_size=params["step_size"],
                regularization=params["regularization"],
                inverse=False,
            ),
        ),
        _step_regularized_spec(
            "inverse_linear",
            history_length,
            lambda dimension, params: LinearRegressionBaseline(
                dimension=dimension,
                step_size=params["step_size"],
                regularization=params["regularization"],
                inverse=True,
            ),
        ),
        _decay_spec("decay_cost", DecayMode.COST),
        _decay_spec("decay_count", DecayMode.COUNT),
        _decay_spec("decay_time", DecayMode.TIME),
        ExperimentModelSpec(
            name="exponential_quadratic",
            input_column="features",
            suggest_params=_suggest_exponential_quadratic_params,
            build_model=lambda params: ExponentialQuadraticBaseline(
                dimension=history_length,
                step_size=params["step_size"],
                regularization=params["regularization"],
            ),
        ),
    ]
    return {spec.name: spec for spec in specs}


def benchmark_model_specs(history_length: int) -> list[ExperimentModelSpec]:
    """Return the maintained benchmark subset of the shared model registry."""
    registry = build_model_registry(history_length)
    return [registry[name] for name in BENCHMARK_MODEL_NAMES]


def single_stream_model_specs(history_length: int) -> dict[str, ExperimentModelSpec]:
    """Return the maintained single-stream subset of the shared model registry."""
    registry = build_model_registry(history_length)
    model_names = BENCHMARK_MODEL_NAMES + SINGLE_STREAM_EXTRA_MODEL_NAMES
    return {name: registry[name] for name in model_names}


def fixed_model_factory(
    spec: ExperimentModelSpec,
    params: Params,
) -> Callable[[], StreamingModelLike]:
    """Build a zero-argument factory from one model spec and fixed parameters."""
    return lambda: spec.build_model(params)


def comparison_model_factories(
    history_length: int,
    step_size: float,
    regularization: float,
) -> dict[str, Callable[[], StreamingModelLike]]:
    """Build the maintained fixed-parameter comparison registry."""
    shared_step_params = {
        "step_size": step_size,
        "regularization": regularization,
    }
    ratio_of_regressors_params = {
        "numerator_step_size": step_size,
        "numerator_regularization": regularization,
        "denominator_step_size": step_size,
        "denominator_regularization": regularization,
        "epsilon": 1e-10,
    }
    registry = build_model_registry(history_length)
    specs = {
        "proximal_softplus": proximal_softplus_spec(),
        "linear_ratio": linear_ratio_spec(),
        "linear_regression": registry["linear"],
        "ratio_of_regressors": registry["ratio_of_regressors"],
    }
    params_by_name = {
        "proximal_softplus": shared_step_params,
        "linear_ratio": shared_step_params,
        "linear_regression": shared_step_params,
        "ratio_of_regressors": ratio_of_regressors_params,
    }
    return {
        name: fixed_model_factory(specs[name], params_by_name[name])
        for name in (
            "proximal_softplus",
            "linear_ratio",
            "linear_regression",
            "ratio_of_regressors",
        )
    }
