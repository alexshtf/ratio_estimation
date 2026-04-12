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
    shifted = np.roll(values, -1)
    shifted[-1] = value
    return shifted


def share_normalizer(numerator: float, denominator: float) -> float:
    """Map a ratio to its numerator share num / (num + den)."""
    value = numerator / (numerator + denominator)
    return value if np.isfinite(value) else np.nan


def log_ratio_normalizer(numerator: float, denominator: float) -> float:
    """Map a positive ratio to log(num / den)."""
    value = np.log(numerator / denominator)
    return float(value) if np.isfinite(value) else np.nan


def inverse_softplus_normalizer(numerator: float, denominator: float) -> float:
    """Map a positive ratio through the inverse of softplus."""
    ratio = (1.0 + numerator) / (1.0 + denominator)
    value = np.log(np.expm1(ratio))
    return float(value) if np.isfinite(value) else np.nan


@dataclass(slots=True)
class RollingMeanWindow:
    """Keep rolling means of the numerator and denominator."""

    window_size: int = 1
    numerator_history: FloatArray = field(init=False, repr=False)
    denominator_history: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.numerator_history = np.zeros(self.window_size, dtype=float)
        self.denominator_history = np.zeros(self.window_size, dtype=float)

    def update(self, numerator: float, denominator: float) -> None:
        """Append one observation to the rolling window."""
        self.numerator_history = shift_left_append(self.numerator_history, numerator)
        self.denominator_history = shift_left_append(self.denominator_history, denominator)

    def mean(self) -> tuple[float, float]:
        """Return the current rolling means."""
        return float(np.mean(self.numerator_history)), float(np.mean(self.denominator_history))


@dataclass(slots=True)
class AutoregressiveRatioFeatures:
    """Store a lagged history of normalized rolling ratios."""

    history_length: int = 24
    window: RollingMeanWindow = field(default_factory=RollingMeanWindow)
    normalizer: RatioNormalizer = inverse_softplus_normalizer
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
        self.ratio_history = shift_left_append(
            self.ratio_history,
            float(np.nan_to_num(normalized_value, nan=0.0)),
        )
        self.missing_history = shift_left_append(
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
