"""Evaluation helpers for streaming ratio experiments."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class StreamingModel(Protocol):
    """A stateful model with predict/update methods for one stream."""

    def predict(self, x: ArrayLike) -> float:
        """Predict one ratio."""
        ...

    def update(self, x: ArrayLike, numerator: float, denominator: float) -> None:
        """Update the model with one observation."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return a lightweight snapshot of the model state."""
        ...


@dataclass(slots=True)
class StreamDiagnostics:
    """A single-stream rollout trace together with the final model state."""

    trace: pd.DataFrame
    final_state: dict[str, Any]

    def tail_mean_log_error(self, tail_fraction: float = 0.5) -> float:
        """Return the mean log error over the tail of the rollout."""
        return tail_mean_log_error(self.trace, tail_fraction=tail_fraction)


def make_json_safe(value: Any) -> Any:
    """Convert arrays and NumPy scalars into JSON-friendly Python values."""
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def log_ratio_error(prediction: float, numerator: float, denominator: float) -> float:
    """Compute the absolute log-ratio error against the observed ratio."""
    safe_prediction = float(np.nextafter(prediction, np.inf))
    safe_numerator = float(np.nextafter(numerator, np.inf))
    safe_denominator = float(np.nextafter(denominator, np.inf))
    return float(abs(np.log(safe_prediction) - np.log(safe_numerator) + np.log(safe_denominator)))


def rollout_stream(
    frame: pd.DataFrame,
    model: StreamingModel,
    input_column: str = "features",
) -> pd.DataFrame:
    """Roll one model forward through a single stream and record predictions."""
    return diagnose_stream(frame, model, input_column=input_column).trace


def diagnose_stream(
    frame: pd.DataFrame,
    model: StreamingModel,
    input_column: str = "features",
) -> StreamDiagnostics:
    """Roll one model through a single stream and capture the final model state."""
    rows: list[dict[str, float]] = []

    for row in frame.to_dict(orient="records"):
        x = row[input_column]
        prediction = float(model.predict(x))
        actual_ratio = float(
            np.nextafter(row["spend"], np.inf) / np.nextafter(row["count"], np.inf)
        )
        rows.append(
            {
                "prediction": prediction,
                "actual_ratio": actual_ratio,
                "true_ratio": row.get("true_ratio", np.nan),
                "log_error": log_ratio_error(prediction, row["spend"], row["count"]),
            }
        )
        model.update(x, row["spend"], row["count"])

    return StreamDiagnostics(
        trace=pd.DataFrame(rows),
        final_state=make_json_safe(model.state_dict()),
    )


def tail_mean_log_error(trace: pd.DataFrame, tail_fraction: float = 0.5) -> float:
    """Return the mean log error over the final tail of a rollout trace."""
    tail_length = max(1, int(np.ceil(len(trace) * tail_fraction)))
    return float(trace["log_error"].tail(tail_length).mean())


def weighted_mean_and_stderr(weights: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return a weighted mean and the Gatz-Smith standard error estimate."""
    mean_value = float(np.average(values, weights=weights))
    n_samples = len(values)
    mean_weight = float(np.mean(weights))
    stderr_squared = (
        n_samples
        * np.sum(np.square(weights) * np.square(values - mean_value))
        / ((n_samples - 1) * (n_samples * mean_weight) ** 2)
    )
    return mean_value, float(np.sqrt(stderr_squared))


def run_panel(
    frame: pd.DataFrame,
    model_factory: Callable[[], StreamingModel],
    input_column: str = "features",
    warmup_steps: int = 2,
    return_stderr: bool = False,
) -> float | tuple[float, float]:
    """Run one model per panel id and return the weighted mean log error."""
    models: dict[int, StreamingModel] = {}
    step_counts = defaultdict(int)
    samples: list[tuple[float, float]] = []

    for row in frame.to_dict(orient="records"):
        model = models.setdefault(int(row["id"]), model_factory())
        x = row[input_column]
        prediction = float(model.predict(x))
        loss = log_ratio_error(prediction, row["spend"], row["count"])

        if step_counts[int(row["id"])] >= warmup_steps:
            samples.append((row["spend"], loss))

        model.update(x, row["spend"], row["count"])
        step_counts[int(row["id"])] += 1

    if not samples:
        return (float("nan"), float("nan")) if return_stderr else float("nan")

    sample_array = np.asarray(samples, dtype=float)
    weights = sample_array[:, 0]
    losses = sample_array[:, 1]
    mean_loss, stderr = weighted_mean_and_stderr(weights, losses)
    return (mean_loss, stderr) if return_stderr else mean_loss
