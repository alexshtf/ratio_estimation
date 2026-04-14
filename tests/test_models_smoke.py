import numpy as np

from ratio_estimation.features import AutoregressiveRatioFeatures
from ratio_estimation.models import (
    LinearInverseRatioLearner,
    LinearRatioLearner,
    RatioProximalLearner,
    SoftplusLink,
)
from ratio_estimation.simulation import sample_campaign


def test_ratio_proximal_learner_runs_on_a_stream() -> None:
    campaign = sample_campaign(rng=np.random.default_rng(0))
    feature_builder = AutoregressiveRatioFeatures(history_length=8)
    learner = RatioProximalLearner(link=SoftplusLink(), step_size=0.1, regularization=1.0)

    predictions = []
    for spend, count in zip(campaign.spend, campaign.count, strict=True):
        x = feature_builder.features()
        predictions.append(learner.predict(x))
        learner.update(x, numerator=spend, denominator=count)
        feature_builder.update(spend, count)

    prediction_array = np.asarray(predictions)
    assert np.all(np.isfinite(prediction_array))
    assert np.all(prediction_array >= 0.0)


def test_linear_ratio_learner_runs_on_a_stream() -> None:
    campaign = sample_campaign(rng=np.random.default_rng(1))
    feature_builder = AutoregressiveRatioFeatures(history_length=6)
    learner = LinearRatioLearner(step_size=0.05, regularization=0.5)

    predictions = []
    for spend, count in zip(campaign.spend[:20], campaign.count[:20], strict=True):
        x = feature_builder.features()
        predictions.append(learner.predict(x))
        learner.update(x, numerator=spend, denominator=count)
        feature_builder.update(spend, count)

    assert np.all(np.isfinite(np.asarray(predictions)))


def test_linear_inverse_ratio_learner_predicts_positive_finite_values() -> None:
    learner = LinearInverseRatioLearner(step_size=0.05, regularization=0.5)
    x = np.array([1.0, -2.0, 0.5])

    cold_start_prediction = learner.predict(x)
    assert np.isfinite(cold_start_prediction)
    assert cold_start_prediction > 0.0

    learner.weights = np.array([-1.0, 0.0, 0.0])
    negative_score_prediction = learner.predict(np.array([1.0, 0.0, 0.0]))
    assert np.isfinite(negative_score_prediction)
    assert negative_score_prediction > 0.0


def test_linear_inverse_ratio_learner_runs_on_a_stream() -> None:
    campaign = sample_campaign(rng=np.random.default_rng(2))
    feature_builder = AutoregressiveRatioFeatures(history_length=6)
    learner = LinearInverseRatioLearner(step_size=0.05, regularization=0.5)

    predictions = []
    for spend, count in zip(campaign.spend[:20], campaign.count[:20], strict=True):
        x = feature_builder.features()
        predictions.append(learner.predict(x))
        learner.update(x, numerator=spend, denominator=count)
        feature_builder.update(spend, count)

    prediction_array = np.asarray(predictions)
    assert np.all(np.isfinite(prediction_array))
    assert np.all(prediction_array > 0.0)
