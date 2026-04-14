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
class PanelLossSamples:
    """Retained post-warmup sample weights and losses from one panel rollout."""

    weights: np.ndarray
    losses: np.ndarray


type PanelProgressCallback = Callable[[int, int], None]


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
    if numerator == 0.0 and denominator == 0.0:
        return float("nan")
    min_positive = float(np.nextafter(0.0, np.inf))
    safe_prediction = max(float(prediction), min_positive)
    safe_numerator = max(float(numerator), min_positive)
    safe_denominator = max(float(denominator), min_positive)
    return float(abs(np.log(safe_prediction) - np.log(safe_numerator) + np.log(safe_denominator)))


def _observed_ratio_array(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Return observed ratios, marking zero/zero rows as undefined."""
    min_positive = float(np.nextafter(0.0, np.inf))
    safe_numerator = np.maximum(np.asarray(numerator, dtype=float), min_positive)
    safe_denominator = np.maximum(np.asarray(denominator, dtype=float), min_positive)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        observed_ratio = safe_numerator / safe_denominator
    undefined_mask = (np.asarray(numerator, dtype=float) == 0.0) & (
        np.asarray(denominator, dtype=float) == 0.0
    )
    observed_ratio[undefined_mask] = np.nan
    return observed_ratio


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
    actual_ratio = _observed_ratio_array(stream.spend, stream.count)

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
    finite_tail_errors = 0

    for index, (x, numerator, denominator) in enumerate(
        zip(stream.inputs, stream.spend, stream.count, strict=True)
    ):
        prediction = float(model.predict(x))
        if index >= tail_start:
            loss = log_ratio_error(prediction, numerator, denominator)
            if np.isfinite(loss):
                tail_error_sum += loss
                finite_tail_errors += 1
        model.update(x, numerator, denominator)

    if finite_tail_errors == 0:
        return float("nan")
    return float(tail_error_sum / finite_tail_errors)


def tail_mean_log_error(trace: pd.DataFrame, tail_fraction: float = 0.5) -> float:
    """Return the mean log error over the final tail of a rollout trace."""
    tail_length = max(1, int(np.ceil(len(trace) * tail_fraction)))
    tail_errors = trace["log_error"].tail(tail_length).to_numpy(dtype=float, copy=False)
    finite_errors = tail_errors[np.isfinite(tail_errors)]
    if len(finite_errors) == 0:
        return float("nan")
    return float(np.mean(finite_errors))


def weighted_mean_and_stderr(weights: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Return a weighted mean and the Gatz-Smith standard error estimate."""
    mean_value = float(np.average(values, weights=weights))
    n_samples = len(values)
    if n_samples <= 1:
        return mean_value, 0.0
    mean_weight = float(np.mean(weights))
    stderr_squared = (
        n_samples
        * np.sum(np.square(weights) * np.square(values - mean_value))
        / ((n_samples - 1) * (n_samples * mean_weight) ** 2)
    )
    return mean_value, float(np.sqrt(stderr_squared))


def summarize_panel_losses(samples: PanelLossSamples) -> tuple[float, float]:
    """Return the weighted mean and standard error for retained panel losses."""
    if len(samples.losses) == 0:
        return float("nan"), float("nan")
    return weighted_mean_and_stderr(samples.weights, samples.losses)


def panel_loss_samples(
    frame: pd.DataFrame,
    model_factory: Callable[[], StreamingModel],
    input_column: str = "features",
    warmup_steps: int = 2,
    progress_callback: PanelProgressCallback | None = None,
    progress_frequency: int = 1_000,
) -> PanelLossSamples:
    """Run one model per panel id and return retained post-warmup loss samples."""
    models: dict[int, StreamingModel] = {}
    step_counts = defaultdict(int)
    ids = frame["id"].to_numpy(dtype=np.int64, copy=False)
    inputs = frame[input_column].to_numpy(copy=False)
    spend = frame["spend"].to_numpy(dtype=float, copy=False)
    count = frame["count"].to_numpy(dtype=float, copy=False)
    sample_weights = np.empty(len(frame), dtype=float)
    sample_losses = np.empty(len(frame), dtype=float)
    n_samples = 0
    total_rows = len(ids)
    callback_frequency = max(1, progress_frequency)

    for row_index, (group_id, x, numerator, denominator) in enumerate(
        zip(ids, inputs, spend, count, strict=True),
        start=1,
    ):
        group_key = int(group_id)
        model = models.get(group_key)
        if model is None:
            model = model_factory()
            models[group_key] = model
        prediction = float(model.predict(x))
        loss = log_ratio_error(prediction, numerator, denominator)

        if step_counts[group_key] >= warmup_steps and np.isfinite(loss):
            sample_weights[n_samples] = numerator
            sample_losses[n_samples] = loss
            n_samples += 1

        model.update(x, numerator, denominator)
        step_counts[group_key] += 1
        if progress_callback is not None and (
            row_index == total_rows or row_index % callback_frequency == 0
        ):
            progress_callback(row_index, total_rows)

    return PanelLossSamples(
        weights=sample_weights[:n_samples].copy(),
        losses=sample_losses[:n_samples].copy(),
    )


def run_panel(
    frame: pd.DataFrame,
    model_factory: Callable[[], StreamingModel],
    input_column: str = "features",
    warmup_steps: int = 2,
    return_stderr: bool = False,
) -> float | tuple[float, float]:
    """Run one model per panel id and return the weighted mean log error."""
    samples = panel_loss_samples(
        frame,
        model_factory=model_factory,
        input_column=input_column,
        warmup_steps=warmup_steps,
    )
    mean_loss, stderr = summarize_panel_losses(samples)
    return (mean_loss, stderr) if return_stderr else mean_loss
