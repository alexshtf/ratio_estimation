import numpy as np
from .proximal_operators import prox_linear_composed, prox_regularized_perturbed, exponential_prox, square_pos_prox, polylog_exp_prox


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
        self.pred_func = pred_func
        self.step_size = step_size
        self.reg_coef = reg_coef
        self.w = None

    def partial_fit(self, x, y_num, y_denom):
        r"""
        Performs one fitting step.
        :param x: The feature vector
        :param y_num: The numerator of the target ratio
        :param y_denom: The denominator of the target ratio
        """
        if self.w is None:
            self.w = np.zeros_like(x)

        def inner_prox(z, eta):
            return prox_linear_composed(z, eta * y_denom, self.pred_func.prox, x, 0)
        self.w = prox_regularized_perturbed(self.w, self.step_size, inner_prox, -y_num * x, self.reg_coef)

    def predict(self, x):
        r"""
        Predicts the ratio of the two quantities for a given feature vector.
        :param x: The feature vector
        :return: The predicted ratio
        """
        return self.pred_func.deriv(np.dot(self.w, x))



class ExponentialPredictor:
    def prox(self, z, eta):
        return exponential_prox(z, eta)

    def deriv(self, z):
        return np.exp(z)


class SquarePosPredictor:
    def prox(self, z, eta):
        return square_pos_prox(z, eta)

    def deriv(self, z):
        return z


class PolylogExpPredictor:
    def prox(self, z, eta):
        return polylog_exp_prox(z, eta)

    def deriv(self, z):
        return np.logaddexp(0, z)

