import numpy as np
from .trends import global_trend, periodic_trend
from .sampling import sample_numerators, sample_denominators

def pos(x):
    return np.logaddexp(0, x)

def combined_trend(hours):
    return pos(global_trend(hours) + periodic_trend(hours))

def sample_campaign(max_offset=24 * 60, mean_length=24 * 14, spend_resolution=25, spend_scale=5, count_shift=0.5):
    offset = np.random.randint(0, 1 + max_offset)
    length = np.random.poisson(mean_length)
    hours = np.arange(offset, offset + length)

    count_trend = count_shift + combined_trend(hours)
    spend_trend = spend_scale * combined_trend(hours) ** 2

    hourly_spend = sample_numerators(spend_trend)
    hourly_count = sample_denominators(spend_resolution * count_trend) / spend_resolution
    true_trend = spend_trend / count_trend

    return hours, true_trend, hourly_spend, hourly_count



