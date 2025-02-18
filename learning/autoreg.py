import numpy as np


def roll_append(arr, x):
    arr = np.roll(arr, -1)
    arr[-1] = x
    return arr


class Window:
    r"""
    Computes the mean ratio over a window of a given length.
    """

    def __init__(self, window_len=1):
        self.numerators = np.zeros(window_len)
        self.denominators = np.zeros(window_len)

    def partial_fit(self, num, denom):
        self.numerators = roll_append(self.numerators, num)
        self.denominators = roll_append(self.denominators, denom)

    def get(self):
        return np.mean(self.numerators), np.mean(self.denominators)


class NormalizedAutoregRatios:
    def __init__(self, window=None, hist_len=24, normalizer=None):
        self.window = window or Window()
        self.normalizer = normalizer or inv_softplus_normalizer
        self.hist = np.zeros(hist_len)
        self.nan_hist = np.zeros(hist_len)

    def partial_fit(self, num, denom):
        self.window.partial_fit(num, denom)
        w_num, w_denom = self.window.get()
        value = self.normalizer(w_num, w_denom)
        self.hist = roll_append(self.hist, np.nan_to_num(value, nan=0))
        self.nan_hist = roll_append(self.nan_hist, np.isnan(value).astype(np.float64))

    def get(self):
        return np.r_[self.hist, self.nan_hist]


def log_logistic_normalizer(num, denom):
    value = num / (num + denom)
    if np.isfinite(value):
        return value
    return np.nan


def log_diff_normalizer(num, denom):
    ratio = num / denom
    value = np.log(ratio)
    if np.isfinite(value):
        return value
    return np.nan


def inv_softplus_normalizer(num, denom):
    ratio = (1 + num) / (1 + denom)
    value = np.log(np.expm1(ratio))
    if np.isfinite(value):
        return value
    return np.nan


class BiasFeature:
    def partial_fit(self, num, denom):
        pass

    def get(self):
        return np.ones(1)


class ConcatFeatures:
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def partial_fit(self, num, denom):
        for f in self.featurizers:
            f.partial_fit(num, denom)

    def get(self):
        features = [f.get() for f in self.featurizers]
        return np.r_[*features]
