import numpy as np
from .proximal_operators import (
    prox_linear_composed,
    prox_regularized_perturbed,
    exponential_prox,
    square_pos_prox,
    square_prox,
    polylog_exp_prox
)


class RatioProximalLearner:
    r"""
    Fits the parameters of a model for predicting the ration of twi quantities via
    .. math::
        \mathrm{ratio}(x) = f'(\langle w, x \rangle)

    by incrementally minimizing the regularized loss
    .. math::
        \ell(w; x,  y_{num}, y_{denom}) =  y_{denom} f(\langle w, x \rangle) - y_{num} \langle w, x \rangle + \frac{\alpha}{2} \|w\|^2

    The losses are minimized using the proximal point algorithm, to ensure stability w.r.t feature
    scaling and step-size selection.
    """
    def __init__(self, pred_func, step_size, reg_coef):
        self._pred_func = pred_func
        self._step_size = step_size
        self._reg_coef = reg_coef
        self.w = None

    def partial_fit(self, x, y_num, y_denom):
        r"""
        Performs one fitting step.
        :param x: The feature vector
        :param y_num: The numerator of the target ratio
        :param y_denom: The denominator of the target ratio
        """
        if not hasattr(self, 'w') or self.w is None:
            self.w = np.zeros_like(x)
        def inner_prox(z, eta):
            return prox_linear_composed(z, eta * y_denom, self._pred_func.prox, x, 0)
        self.w = prox_regularized_perturbed(self.w, self._step_size, inner_prox, -y_num * x, self._reg_coef)

    def predict(self, x):
        r"""
        Predicts the ratio of the two quantities for a given feature vector.
        :param x: The feature vector
        :return: The predicted ratio
        """
        return self._pred_func.deriv(np.dot(self.w, x))



class ExponentialPredictor:
    r"""
    A predictor for the ratio of two quantities of the form :math:`\exp(\langle w, x \rangle)`.
    """
    def prox(self, z, eta):
        return exponential_prox(z, eta)

    def deriv(self, z):
        return np.exp(z)


class SquarePosPredictor:
    r"""
    A predictor for the ratio of two quantities of the form :math:`\max(0, \langle w, x \rangle)`.
    """
    def prox(self, z, eta):
        return square_pos_prox(z, eta)

    def deriv(self, z):
        return np.maximum(0, z)


class PolylogExpPredictor:
    r"""
    A predictor for the ratio of two quantities of the form :math:`\ln(1+\exp(\langle w, x \rangle))`.
    """
    def prox(self, z, eta):
        return polylog_exp_prox(z, eta)

    def deriv(self, z):
        return np.logaddexp(0, z)


class LinRegRatioLearner:
    r"""
    Fits a linear ratio predictor :math:`\langle x, w \rangle` using least-squares regression, by minimizing a regularized loss
    of the form:
    .. math::
        \ell(w; x, y_{num}, y_{denom}) = \frac{1}{2}(y_{denom} \langle w, x \rangle - y_{num})^2 + \frac{\alpha}{2} \|w\|^2
    """
    def __init__(self, step_size, reg_coef):
        self._step_size = step_size
        self._reg_coef = reg_coef
        self.w = None

    def partial_fit(self, x, y_num, y_denom):
        r"""
        Performs one fitting step.
        :param x: The feature vector
        :param y_num: The numerator of the target ratio
        :param y_denom: The denominator of the target ratio
        """
        if not hasattr(self, 'w') or self.w is None:
            self.w = np.zeros_like(x)

        def inner_prox(z, eta):
            return prox_linear_composed(z, eta, square_prox, y_denom * x, 0)
        self.w = prox_regularized_perturbed(self.w, self._step_size, inner_prox, -y_num * y_denom * x, self._reg_coef)

    def predict(self, x):
        r"""
        Predicts the ratio of the two quantities for a given feature vector.
        :param x: The feature vector
        :return: The predicted ratio
        """
        return np.dot(self.w, x)


class LinRegInvRatioLearner:
    r"""
    Fits a linear inverse ratio predictor :math:`\frac{1}{\langle x, w \rangle}` using least-squares regression, by minimizing a regularized loss
    of the form:
    .. math::
        \ell(w; x, y_{num}, y_{denom}) = \frac{1}{2}(y_{num} \langle w, x \rangle - y_{denom})^2 + \frac{\alpha}{2} \|w\|^2
    """

    def __init__(self, step_size, reg_coef):
        self._step_size = step_size
        self._reg_coef = reg_coef

    def partial_fit(self, x, y_num, y_denom):
        r"""
        Performs one fitting step.
        :param x: The feature vector
        :param y_num: The numerator of the target ratio
        :param y_denom: The denominator of the target ratio
        """
        if not hasattr(self, 'w') or self.w is None:
            self.w = np.zeros_like(x)

        def inner_prox(z, eta):
            return prox_linear_composed(z, eta, square_prox, y_num * x, 0)
        self.w = prox_regularized_perturbed(self.w, self._step_size, inner_prox, -y_num * y_denom * x, self._reg_coef)

    def predict(self, x):
        r"""
        Predicts the ratio of the two quantities for a given feature vector.
        :param x: The feature vector
        :return: The predicted ratio
        """
        return np.reciprocal(np.dot(self.w, x))