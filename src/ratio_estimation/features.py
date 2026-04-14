"""Autoregressive feature builders for online ratio estimation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RatioNormalizer = Callable[[float, float], float]


class FeatureBlock(Protocol):
    """A stateful block that emits one feature vector."""

    def update(self, numerator: float, denominator: float) -> None:
        """Update the block with one observation."""
        ...

    def features(self) -> FloatArray:
        """Return the current feature vector."""
        ...


def shift_left_append(values: FloatArray, value: float) -> FloatArray:
    """Shift a 1D array left and append one new value."""
    shifted = np.empty_like(values)
    if len(values) > 1:
        shifted[:-1] = values[1:]
    shifted[-1] = value
    return shifted


def _shift_left_append_inplace(values: FloatArray, value: float) -> None:
    """Shift a 1D array left in place and append one new value."""
    if len(values) > 1:
        values[:-1] = values[1:]
    values[-1] = value


def share_normalizer(numerator: float, denominator: float) -> float:
    """Map a ratio to its numerator share num / (num + den)."""
    value = numerator / (numerator + denominator)
    return value if np.isfinite(value) else np.nan


def log_ratio_normalizer(numerator: float, denominator: float) -> float:
    """Map a positive ratio to log(num / den)."""
    value = np.log(numerator / denominator)
    return float(value) if np.isfinite(value) else np.nan


def smoothed_inverse_softplus_normalizer(numerator: float, denominator: float) -> float:
    """Map a smoothed positive ratio through an inverse-softplus-style transform."""
    ratio = (1.0 + numerator) / (1.0 + denominator)
    value = np.log(np.expm1(ratio))
    return float(value) if np.isfinite(value) else np.nan


def inverse_softplus_normalizer(numerator: float, denominator: float) -> float:
    """Backward-compatible alias for the smoothed inverse-softplus normalizer."""
    return smoothed_inverse_softplus_normalizer(numerator, denominator)


@dataclass(slots=True)
class RollingMeanWindow:
    """Keep rolling means of the numerator and denominator."""

    window_size: int = 1
    numerator_history: FloatArray = field(init=False, repr=False)
    denominator_history: FloatArray = field(init=False, repr=False)
    numerator_sum: float = field(init=False, repr=False)
    denominator_sum: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.numerator_history = np.zeros(self.window_size, dtype=float)
        self.denominator_history = np.zeros(self.window_size, dtype=float)
        self.numerator_sum = 0.0
        self.denominator_sum = 0.0

    def update(self, numerator: float, denominator: float) -> None:
        """Append one observation to the rolling window."""
        oldest_numerator = float(self.numerator_history[0])
        oldest_denominator = float(self.denominator_history[0])
        _shift_left_append_inplace(self.numerator_history, numerator)
        _shift_left_append_inplace(self.denominator_history, denominator)
        self.numerator_sum += numerator - oldest_numerator
        self.denominator_sum += denominator - oldest_denominator

    def mean(self) -> tuple[float, float]:
        """Return the current rolling means."""
        return (
            float(self.numerator_sum / self.window_size),
            float(self.denominator_sum / self.window_size),
        )


@dataclass(slots=True)
class AutoregressiveRatioFeatures:
    """Store a lagged history of normalized rolling ratios."""

    history_length: int = 24
    window: RollingMeanWindow = field(default_factory=RollingMeanWindow)
    normalizer: RatioNormalizer = smoothed_inverse_softplus_normalizer
    ratio_history: FloatArray = field(init=False, repr=False)
    missing_history: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ratio_history = np.zeros(self.history_length, dtype=float)
        self.missing_history = np.zeros(self.history_length, dtype=float)

    def update(self, numerator: float, denominator: float) -> None:
        """Append one observation to the autoregressive history."""
        self.window.update(numerator, denominator)
        mean_numerator, mean_denominator = self.window.mean()
        normalized_value = self.normalizer(mean_numerator, mean_denominator)
        _shift_left_append_inplace(
            self.ratio_history,
            float(np.nan_to_num(normalized_value, nan=0.0)),
        )
        _shift_left_append_inplace(
            self.missing_history,
            float(np.isnan(normalized_value)),
        )

    def features(self) -> FloatArray:
        """Return the current feature vector."""
        return np.concatenate([self.ratio_history, self.missing_history])


class BiasFeature:
    """Return a constant one-dimensional bias feature."""

    def update(self, numerator: float, denominator: float) -> None:
        """Ignore streaming updates."""
        _ = numerator, denominator

    def features(self) -> FloatArray:
        """Return the constant bias feature."""
        return np.ones(1, dtype=float)


class FeatureStack:
    """Concatenate several feature blocks into one feature vector."""

    def __init__(self, *feature_blocks: FeatureBlock) -> None:
        self.feature_blocks = feature_blocks

    def update(self, numerator: float, denominator: float) -> None:
        """Update all feature blocks with one observation."""
        for feature_block in self.feature_blocks:
            feature_block.update(numerator, denominator)

    def features(self) -> FloatArray:
        """Return the concatenated feature vector."""
        return np.concatenate([feature_block.features() for feature_block in self.feature_blocks])
