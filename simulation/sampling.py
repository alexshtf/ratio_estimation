import numpy as np

def sample_denominators(trend):
    return np.random.poisson(trend)

def sample_numerators(trend, dispersion=0.75):
    variance = 1 + dispersion * np.square(trend)
    p = trend / variance
    n = np.square(trend) / (variance - trend)
    return np.random.negative_binomial(n, p)