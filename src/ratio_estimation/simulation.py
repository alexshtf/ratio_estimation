"""Synthetic campaign simulation utilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import binom

FloatArray = NDArray[np.float64]


@dataclass(slots=True)
class CampaignSample:
    """One simulated campaign trajectory."""

    hours: NDArray[np.int64]
    true_ratio: FloatArray
    spend: FloatArray
    count: NDArray[np.int64]


def softplus(x: ArrayLike) -> FloatArray:
    """Apply softplus elementwise."""
    return np.logaddexp(0.0, np.asarray(x, dtype=float))


def periodic_trend(
    hours: ArrayLike,
    coef_df: float = 3.0,
    max_periods: int = 2,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Sample a random daily periodic trend with the maintained benchmark semantics."""
    generator = np.random.default_rng() if rng is None else rng
    hour_array = np.asarray(hours, dtype=float)
    angles = (2.0 * np.pi * hour_array / 24.0)[None, :]
    num_periods = 1 if max_periods <= 1 else int(generator.integers(1, max_periods))
    coefficients = generator.standard_t(df=coef_df, size=(num_periods, 1))
    phases = generator.uniform(0.0, 2.0 * np.pi, size=(num_periods, 1))
    frequencies = np.arange(1, num_periods + 1, dtype=float)[:, None]
    return np.sum(coefficients * np.cos(frequencies * angles + phases), axis=0)


def global_trend(
    hours: ArrayLike,
    coef_df: float = 3.0,
    num_coefficients: int = 4,
    bias: float = 0.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Sample a smooth global trend over the full campaign horizon."""
    generator = np.random.default_rng() if rng is None else rng
    hour_array = np.asarray(hours, dtype=float)
    if hour_array.size == 1:
        normalized_hours = np.zeros(1, dtype=float)
    else:
        normalized_hours = (hour_array - hour_array.min()) / (hour_array.max() - hour_array.min())
    coefficients = bias + generator.standard_t(df=coef_df, size=num_coefficients)
    basis = binom.pmf(
        np.arange(num_coefficients, dtype=int),
        num_coefficients - 1,
        normalized_hours[:, None],
    )
    return basis @ coefficients


def combined_trend(hours: ArrayLike, rng: np.random.Generator | None = None) -> FloatArray:
    """Combine the global and periodic trends and map them to positive values."""
    generator = np.random.default_rng() if rng is None else rng
    latent_signal = global_trend(hours, rng=generator) + periodic_trend(hours, rng=generator)
    return softplus(latent_signal)


def sample_poisson(mean: ArrayLike, rng: np.random.Generator | None = None) -> NDArray[np.int64]:
    """Sample counts from a Poisson model."""
    generator = np.random.default_rng() if rng is None else rng
    return generator.poisson(np.asarray(mean, dtype=float))


def sample_negative_binomial(
    mean: ArrayLike,
    dispersion: float = 0.75,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sample overdispersed counts with the maintained quadratic-variance law."""
    generator = np.random.default_rng() if rng is None else rng
    mean_array = np.asarray(mean, dtype=float)
    variance = 1.0 + dispersion * np.square(mean_array)
    probability = mean_array / variance
    num_failures = np.square(mean_array) / (variance - mean_array)
    return generator.negative_binomial(num_failures, probability)


def sample_campaign(
    max_offset: int = 24 * 60,
    mean_length: int = 24 * 14,
    spend_resolution: int = 25,
    spend_scale: float = 5.0,
    count_shift: float = 0.5,
    rng: np.random.Generator | None = None,
) -> CampaignSample:
    """Sample one campaign with independent latent spend and count trends."""
    generator = np.random.default_rng() if rng is None else rng
    offset = int(generator.integers(0, max_offset + 1))
    length = max(1, int(generator.poisson(mean_length)))
    hours = np.arange(offset, offset + length, dtype=np.int64)

    count_latent = combined_trend(hours, rng=generator)
    spend_latent = combined_trend(hours, rng=generator)
    count_trend = count_shift + count_latent
    spend_trend = spend_scale * np.square(spend_latent)

    spend = (
        sample_negative_binomial(spend_resolution * spend_trend, rng=generator) / spend_resolution
    )
    count = sample_poisson(count_trend, rng=generator)
    true_ratio = spend_trend / count_trend

    return CampaignSample(hours=hours, true_ratio=true_ratio, spend=spend, count=count)
