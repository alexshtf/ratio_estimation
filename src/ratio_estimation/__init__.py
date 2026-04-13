"""Small research library for online ratio estimation."""

from .features import (
    AutoregressiveRatioFeatures,
    BiasFeature,
    FeatureStack,
    RollingMeanWindow,
    inverse_softplus_normalizer,
    log_ratio_normalizer,
    share_normalizer,
)
from .models import (
    ExponentialLink,
    LinearInverseRatioLearner,
    LinearRatioLearner,
    PositivePartLink,
    RatioProximalLearner,
    SoftplusLink,
)
from .simulation import (
    CampaignSample,
    combined_trend,
    global_trend,
    periodic_trend,
    sample_campaign,
    sample_negative_binomial,
    sample_poisson,
)

__all__ = [
    "AutoregressiveRatioFeatures",
    "BiasFeature",
    "CampaignSample",
    "ExponentialLink",
    "FeatureStack",
    "LinearInverseRatioLearner",
    "LinearRatioLearner",
    "PositivePartLink",
    "RatioProximalLearner",
    "RollingMeanWindow",
    "SoftplusLink",
    "combined_trend",
    "global_trend",
    "inverse_softplus_normalizer",
    "log_ratio_normalizer",
    "periodic_trend",
    "sample_campaign",
    "sample_negative_binomial",
    "sample_poisson",
    "share_normalizer",
]
