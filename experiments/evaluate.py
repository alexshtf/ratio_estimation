"""Evaluation helpers for streaming ratio experiments."""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .io import make_json_safe


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


@dataclass(slots=True)
class _StreamArrays:
    """Array-backed columns for one streaming evaluation loop."""

    inputs: np.ndarray
    spend: np.ndarray
    count: np.ndarray
    true_ratio: np.ndarray


def _stream_arrays(frame: pd.DataFrame, input_column: str) -> _StreamArrays:
    """Extract the columns needed by one streaming evaluation loop."""
    true_ratio = (
        frame["true_ratio"].to_numpy(dtype=float, copy=False)
        if "true_ratio" in frame
        else np.full(len(frame), np.nan, dtype=float)
    )
    return _StreamArrays(
        inputs=frame[input_column].to_numpy(copy=False),
        spend=frame["spend"].to_numpy(dtype=float, copy=False),
        count=frame["count"].to_numpy(dtype=float, copy=False),
        true_ratio=true_ratio,
    )


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
    stream = _stream_arrays(frame, input_column)
    n_rows = len(stream.spend)
    predictions = np.empty(n_rows, dtype=float)
    log_errors = np.empty(n_rows, dtype=float)
    actual_ratio = np.nextafter(stream.spend, np.inf) / np.nextafter(stream.count, np.inf)

    for index, (x, numerator, denominator) in enumerate(
        zip(stream.inputs, stream.spend, stream.count, strict=True)
    ):
        prediction = float(model.predict(x))
        predictions[index] = prediction
        log_errors[index] = log_ratio_error(prediction, numerator, denominator)
        model.update(x, numerator, denominator)

    return StreamDiagnostics(
        trace=pd.DataFrame(
            {
                "prediction": predictions,
                "actual_ratio": actual_ratio,
                "true_ratio": stream.true_ratio,
                "log_error": log_errors,
            }
        ),
        final_state=make_json_safe(model.state_dict()),
    )


def score_stream_tail(
    frame: pd.DataFrame,
    model: StreamingModel,
    input_column: str = "features",
    tail_fraction: float = 0.5,
) -> float:
    """Roll one model through a single stream and return only the tail mean log error."""
    stream = _stream_arrays(frame, input_column)
    n_rows = len(stream.spend)
    if n_rows == 0:
        return float("nan")

    tail_length = max(1, int(np.ceil(n_rows * tail_fraction)))
    tail_start = n_rows - tail_length
    tail_error_sum = 0.0

    for index, (x, numerator, denominator) in enumerate(
        zip(stream.inputs, stream.spend, stream.count, strict=True)
    ):
        prediction = float(model.predict(x))
        if index >= tail_start:
            tail_error_sum += log_ratio_error(prediction, numerator, denominator)
        model.update(x, numerator, denominator)

    return float(tail_error_sum / tail_length)


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
    ids = frame["id"].to_numpy(dtype=np.int64, copy=False)
    inputs = frame[input_column].to_numpy(copy=False)
    spend = frame["spend"].to_numpy(dtype=float, copy=False)
    count = frame["count"].to_numpy(dtype=float, copy=False)
    sample_weights = np.empty(len(frame), dtype=float)
    sample_losses = np.empty(len(frame), dtype=float)
    n_samples = 0

    for group_id, x, numerator, denominator in zip(ids, inputs, spend, count, strict=True):
        group_key = int(group_id)
        model = models.setdefault(group_key, model_factory())
        prediction = float(model.predict(x))
        loss = log_ratio_error(prediction, numerator, denominator)

        if step_counts[group_key] >= warmup_steps:
            sample_weights[n_samples] = numerator
            sample_losses[n_samples] = loss
            n_samples += 1

        model.update(x, numerator, denominator)
        step_counts[group_key] += 1

    if n_samples == 0:
        return (float("nan"), float("nan")) if return_stderr else float("nan")

    weights = sample_weights[:n_samples]
    losses = sample_losses[:n_samples]
    mean_loss, stderr = weighted_mean_and_stderr(weights, losses)
    return (mean_loss, stderr) if return_stderr else mean_loss
