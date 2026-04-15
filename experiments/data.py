"""Dataset builders used by the experiment notebooks and tuning scripts."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ratio_estimation.simulation import sample_negative_binomial, sample_poisson


@dataclass(slots=True)
class _LatentAdGroupPaths:
    """Latent bounded paths used to generate one synthetic ad group."""

    offset_series: np.ndarray
    spend_mean: np.ndarray
    count_mean: np.ndarray
    true_ratio: np.ndarray


def bounded_periodic_series(
    lower: float,
    upper: float,
    n_samples: int = 48,
    frequencies: int | np.ndarray = 2,
    phases: float | np.ndarray = np.pi / 4,
    noise_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a smooth positive series constrained to the interval [lower, upper]."""
    generator = np.random.default_rng() if rng is None else rng
    frequency_array = np.atleast_1d(np.asarray(frequencies, dtype=float))
    phase_array = np.atleast_1d(np.asarray(phases, dtype=float))

    log_scale = 0.5 * (np.log(upper) - np.log(lower))
    log_shift = 0.5 * (np.log(upper) + np.log(lower))

    samples = np.ones(n_samples, dtype=float)
    for frequency, phase in zip(frequency_array, phase_array, strict=True):
        angles = np.linspace(0.0, frequency * 2.0 * np.pi, n_samples) + phase
        latent_signal = generator.normal(loc=np.cos(angles), scale=noise_scale)
        latent_signal = np.clip(latent_signal, -1.0, 1.0)
        samples *= np.exp(log_scale * latent_signal + log_shift)
    return np.clip(np.power(samples, 1.0 / len(frequency_array)), lower, upper)


def _sample_ad_group_latent_paths(
    mean_spend: float = 100.0,
    mean_ratio: float = 5.0,
    mean_samples: float = 24 * 7,
    mean_frequency: float = 1.5,
    num_frequencies: int = 2,
    max_time_offset: int = 24 * 9,
    rng: np.random.Generator | None = None,
) -> _LatentAdGroupPaths:
    """Sample bounded latent spend and ratio paths for one synthetic ad group."""
    generator = np.random.default_rng() if rng is None else rng
    if max_time_offset < 2:
        raise ValueError("max_time_offset must be at least 2 for the experiment generator.")

    n_samples = max_time_offset
    while n_samples + 1 > max_time_offset:
        n_samples = int(generator.poisson(mean_samples)) + 1

    frequencies = generator.poisson(mean_frequency, num_frequencies) + 1
    phases = generator.uniform(0.0, 2.0 * np.pi, num_frequencies)
    offset = int(generator.integers(0, max_time_offset - n_samples))
    min_spend, max_spend = np.sort(generator.exponential(mean_spend, 2))
    min_ratio, max_ratio = np.sort(generator.exponential(mean_ratio, 2))

    spend_mean = bounded_periodic_series(
        min_spend,
        max_spend,
        n_samples=n_samples,
        frequencies=frequencies,
        phases=phases,
        rng=generator,
    )
    true_ratio = bounded_periodic_series(
        min_ratio,
        max_ratio,
        n_samples=n_samples,
        frequencies=frequencies,
        phases=phases,
        rng=generator,
    )
    count_mean = spend_mean / true_ratio
    offset_series = offset + np.arange(n_samples, dtype=int)
    return _LatentAdGroupPaths(
        offset_series=offset_series,
        spend_mean=spend_mean,
        count_mean=count_mean,
        true_ratio=true_ratio,
    )


def sample_ad_group(
    group_id: int = 0,
    mean_spend: float = 100.0,
    mean_ratio: float = 5.0,
    mean_samples: float = 24 * 7,
    mean_frequency: float = 1.5,
    num_frequencies: int = 2,
    max_time_offset: int = 24 * 9,
    spend_resolution: int = 25,
    spend_dispersion: float = 0.75,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample one synthetic ad group from latent bounded means and stochastic observations.

    The experiment generator requires `max_time_offset >= 2`.
    """
    generator = np.random.default_rng() if rng is None else rng
    latent_paths = _sample_ad_group_latent_paths(
        mean_spend=mean_spend,
        mean_ratio=mean_ratio,
        mean_samples=mean_samples,
        mean_frequency=mean_frequency,
        num_frequencies=num_frequencies,
        max_time_offset=max_time_offset,
        rng=generator,
    )
    spend = (
        sample_negative_binomial(
            spend_resolution * latent_paths.spend_mean,
            dispersion=spend_dispersion,
            rng=generator,
        )
        / spend_resolution
    )
    count = sample_poisson(latent_paths.count_mean, rng=generator)

    return pd.DataFrame(
        {
            "id": group_id,
            "offset": latent_paths.offset_series,
            "spend": spend,
            "count": count,
            "true_ratio": latent_paths.true_ratio,
        }
    )


def add_autoregressive_features(frame: pd.DataFrame, history_length: int = 3) -> pd.DataFrame:
    """Add padded rolling ratio-share features from previous observations only."""
    if frame.empty:
        result = frame.copy()
        result["features"] = pd.Series(dtype=object)
        return result

    spend = frame["spend"].to_numpy(dtype=float)
    count = frame["count"].to_numpy(dtype=float)
    total = spend + count
    ratio_share = np.divide(
        spend,
        total,
        out=np.zeros_like(spend, dtype=float),
        where=total > 0.0,
    )
    lagged_share = np.pad(ratio_share, (history_length, 0))[:-1]
    feature_matrix = sliding_window_view(lagged_share, history_length).copy()

    result = frame.copy()
    result["features"] = list(feature_matrix)
    return result


def generate_dataset(
    n_groups: int,
    history_length: int = 3,
    rng: np.random.Generator | None = None,
    **group_kwargs: Any,
) -> pd.DataFrame:
    """Generate a multi-group panel with causal lagged ratio-share experiment features."""
    generator = np.random.default_rng() if rng is None else rng
    frames = [
        add_autoregressive_features(
            sample_ad_group(group_id=group_id, rng=generator, **group_kwargs),
            history_length=history_length,
        )
        for group_id in range(n_groups)
    ]
    dataset = pd.concat(frames, ignore_index=True).sort_values("offset", kind="stable")
    dataset = dataset.reset_index(drop=True)
    dataset["offset"] -= int(dataset["offset"].min())
    return dataset
