import numpy as np

from ratio_estimation.features import (
    AutoregressiveRatioFeatures,
    BiasFeature,
    FeatureStack,
    RollingMeanWindow,
    share_normalizer,
)


def test_rolling_mean_window_tracks_recent_values() -> None:
    window = RollingMeanWindow(window_size=3)
    window.update(1.0, 2.0)
    window.update(4.0, 5.0)
    window.update(7.0, 8.0)
    assert window.mean() == (4.0, 5.0)
    window.update(10.0, 11.0)
    assert window.mean() == (7.0, 8.0)


def test_autoregressive_features_store_ratio_history() -> None:
    features = AutoregressiveRatioFeatures(history_length=3, normalizer=share_normalizer)
    np.testing.assert_allclose(features.features(), np.zeros(6))

    features.update(1.0, 1.0)
    np.testing.assert_allclose(features.features()[:3], [0.0, 0.0, 0.5])
    np.testing.assert_allclose(features.features()[3:], [0.0, 0.0, 0.0])

    features.update(3.0, 1.0)
    np.testing.assert_allclose(features.features()[:3], [0.0, 0.5, 0.75])


def test_feature_stack_concatenates_blocks() -> None:
    stacked = FeatureStack(
        AutoregressiveRatioFeatures(history_length=2, normalizer=share_normalizer),
        BiasFeature(),
    )
    stacked.update(2.0, 2.0)
    assert stacked.features().shape == (5,)
