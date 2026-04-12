import numpy as np
import pytest
from numpy.typing import NDArray

import ratio_estimation.simulation as simulation
from ratio_estimation.simulation import (
    global_trend,
    periodic_trend,
    sample_campaign,
    sample_negative_binomial,
)


def test_trend_shapes_match_hours() -> None:
    hours = np.arange(48)
    assert periodic_trend(hours, rng=np.random.default_rng(0)).shape == hours.shape
    assert global_trend(hours, rng=np.random.default_rng(0)).shape == hours.shape


def test_negative_binomial_samples_are_nonnegative() -> None:
    samples = sample_negative_binomial([1.0, 2.0, 3.0], rng=np.random.default_rng(0))
    assert np.all(samples >= 0)


def test_negative_binomial_uses_benchmark_variance_law() -> None:
    mean = 3.0
    dispersion = 0.75
    draws = sample_negative_binomial(
        np.full(50_000, mean),
        dispersion=dispersion,
        rng=np.random.default_rng(0),
    )
    expected_variance = 1.0 + dispersion * mean**2
    assert abs(np.var(draws) - expected_variance) < 0.2


def test_periodic_trend_matches_benchmark_single_harmonic_default() -> None:
    hours = np.arange(6)
    observed = periodic_trend(hours, max_periods=2, rng=np.random.default_rng(0))

    manual_rng = np.random.default_rng(0)
    _ = manual_rng.integers(1, 2)
    coefficients = manual_rng.standard_t(df=3.0, size=(1, 1))
    phases = manual_rng.uniform(0.0, 2.0 * np.pi, size=(1, 1))
    angles = (2.0 * np.pi * hours / 24.0)[None, :]
    expected = np.sum(coefficients * np.cos(angles + phases), axis=0)

    np.testing.assert_allclose(observed, expected)


def test_sample_campaign_uses_separate_spend_and_count_latents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trends = iter(
        [
            np.array([1.0], dtype=float),
            np.array([4.0], dtype=float),
        ]
    )

    def fake_combined_trend(
        hours: NDArray[np.int64],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        _ = hours, rng
        return next(trends)

    monkeypatch.setattr(simulation, "combined_trend", fake_combined_trend)
    campaign = simulation.sample_campaign(max_offset=0, mean_length=0, rng=np.random.default_rng(0))

    np.testing.assert_allclose(campaign.true_ratio, [80.0 / 1.5])


def test_campaign_sampling_is_reproducible() -> None:
    first = sample_campaign(rng=np.random.default_rng(0))
    second = sample_campaign(rng=np.random.default_rng(0))

    assert np.array_equal(first.hours, second.hours)
    assert np.allclose(first.true_ratio, second.true_ratio)
    assert np.allclose(first.spend, second.spend)
    assert np.array_equal(first.count, second.count)
    assert np.all(first.spend >= 0)
    assert np.all(first.count >= 0)
