"""Dataset builders used by the experiment notebooks and tuning scripts."""

from typing import Any

import numpy as np
import pandas as pd


def bounded_periodic_series(
    lower: float,
    upper: float,
    n_samples: int = 48,
    frequencies: int | np.ndarray = 2,
    phases: float | np.ndarray = np.pi / 4,
    noise_scale: float = 0.05,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a smooth positive series bounded between two random scales."""
    generator = np.random.default_rng() if rng is None else rng
    frequency_array = np.atleast_1d(np.asarray(frequencies, dtype=float))
    phase_array = np.atleast_1d(np.asarray(phases, dtype=float))

    log_scale = 0.5 * (np.log(upper) - np.log(lower))
    log_shift = 0.5 * (np.log(upper) + np.log(lower))

    samples = np.ones(n_samples, dtype=float)
    for frequency, phase in zip(frequency_array, phase_array, strict=True):
        angles = np.linspace(0.0, frequency * 2.0 * np.pi, n_samples) + phase
        latent_signal = generator.normal(loc=np.cos(angles), scale=noise_scale)
        samples *= np.exp(log_scale * latent_signal + log_shift)
    return np.power(samples, 1.0 / len(frequency_array))


def sample_ad_group(
    group_id: int = 0,
    mean_spend: float = 100.0,
    mean_ratio: float = 5.0,
    mean_samples: float = 24 * 7,
    mean_frequency: float = 1.5,
    num_frequencies: int = 2,
    max_time_offset: int = 24 * 9,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Sample one synthetic ad group with spend, count, and true ratio."""
    generator = np.random.default_rng() if rng is None else rng

    n_samples = max_time_offset
    while n_samples + 1 > max_time_offset:
        n_samples = int(generator.poisson(mean_samples)) + 1

    frequencies = generator.poisson(mean_frequency, num_frequencies) + 1
    phases = generator.uniform(0.0, 2.0 * np.pi, num_frequencies)
    offset = int(generator.integers(0, max_time_offset - n_samples))
    min_spend, max_spend = np.sort(generator.exponential(mean_spend, 2))
    min_ratio, max_ratio = np.sort(generator.exponential(mean_ratio, 2))

    spend = bounded_periodic_series(
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
    count = np.asarray(spend / true_ratio, dtype=int)
    offset_series = offset + np.arange(n_samples, dtype=int)

    return pd.DataFrame(
        {
            "id": group_id,
            "offset": offset_series,
            "spend": spend,
            "count": count,
            "true_ratio": true_ratio,
        }
    )


def add_autoregressive_features(frame: pd.DataFrame, history_length: int = 3) -> pd.DataFrame:
    """Add padded rolling ratio-share features, including the current observation."""
    spend = frame["spend"].to_numpy(dtype=float)
    count = frame["count"].to_numpy(dtype=float)
    ratio_share = spend / (spend + count)
    features = [
        np.pad(
            ratio_share[max(0, index - history_length + 1) : index + 1],
            (max(0, history_length - index - 1), 0),
        )
        for index in range(len(frame))
    ]

    result = frame.copy()
    result["features"] = features
    return result


def generate_dataset(
    n_groups: int,
    history_length: int = 3,
    rng: np.random.Generator | None = None,
    **group_kwargs: Any,
) -> pd.DataFrame:
    """Generate a multi-group panel with the maintained benchmark feature semantics."""
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
