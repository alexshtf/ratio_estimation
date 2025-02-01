import numpy as np
import scipy


def periodic_trend(hours, coef_t_df=3, max_num_periods=2):
    twopi = 2 * np.pi
    angles = twopi * hours / 24 # arg for periodic trends: day = 2pi
    angles = angles.reshape(1, -1)

    # Fourier sequence of num_of_periods frequencies - periodic trend
    num_of_periods = np.random.randint(1, max_num_periods)
    coefficients = np.random.standard_t(df=coef_t_df, size=(num_of_periods, 1))
    phases = np.random.uniform(low=0, high=twopi, size=(num_of_periods, 1))
    frequencies = np.arange(1, 1 + num_of_periods).reshape(num_of_periods, 1)
    return np.sum(
        coefficients * np.cos(frequencies * angles + phases),
        axis=0
    )


def global_trend(hours, coef_t_df=3, num_coefs=4, bias=0):
    max_hour = np.max(hours)
    min_hour = np.min(hours)
    normalized_hours = (hours - min_hour) / (max_hour - min_hour)

    # Bezier curve with 3 coefficients - global trend
    #    see: https://en.wikipedia.org/wiki/B%C3%A9zier_curve
    global_trend_coef = bias + np.random.standard_t(df=coef_t_df, size=num_coefs)
    global_trend_basis = scipy.stats.binom.pmf(
        np.arange(num_coefs), num_coefs - 1, normalized_hours.reshape(-1, 1))
    return global_trend_basis @ global_trend_coef