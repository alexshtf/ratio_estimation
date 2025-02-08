import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Callable
from scipy.special import wrightomega, expit


def prox_regularized_perturbed(
        x: ArrayLike,
        eta: ArrayLike | float,
        inner_prox: Callable[[ArrayLike, ArrayLike | float], ArrayLike],
        phi: ArrayLike,
        alpha: Union[ArrayLike, float]):
    r"""
    Proximal operator of a regularized-perturbed function of the form
    .. math::
        f(x; \phi, alpha) = h(x) + \langle \phi, x \rangle + \frac{\alpha}{2} \|x\|^2

    :param x: The point at which to evaluate the proximal operator
    :param eta: The step-size
    :param inner_prox: The proximal operator of :math:`h(x)`
    :param phi: The linear perturbation coefficients
    :param alpha: The regularization coefficient.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    x = np.asarray(x)
    phi = np.asarray(phi)
    if not np.isscalar(eta):
        eta = np.asarray(eta)[..., np.newaxis]
    if not np.isscalar(alpha):
        alpha = np.asarray(alpha)[..., np.newaxis]


    inner_step_size = eta / (1 + eta * alpha)
    if not np.isscalar(inner_step_size):
        inner_step_size = np.squeeze(inner_step_size, axis=-1)

    inner_prox_arg = (x - eta * phi) / (1 + eta * alpha)
    return inner_prox(inner_prox_arg, inner_step_size)


def prox_linear_composed(
        x: ArrayLike,
        eta: ArrayLike | float,
        inner_prox: Callable[[ArrayLike | float, ArrayLike | float], ArrayLike | float],
        theta: ArrayLike,
        b: ArrayLike | float):
    r"""
    Proximal operator of a the composition of a univariate function onto an affine function, namely,
    .. math::
        f(x; \theta, b) = g(\langle \theta, x \rangle + b),
    where :math:`g(z)` is a univariate function with a known proximal operator.

    :param x: The point at which we compute the proximal operator.
    :param eta: The step-size
    :param inner_prox: The proximal operator of the scalar function :math:`g(z)`.
    :param theta: The :math:`theta` vector from the definition of :math:`f` above.
    :param b: The :math:`b` scalar from the definition of :math:`f` above.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    x = np.asarray(x)
    theta = np.asarray(theta)
    if not np.isscalar(b):
        b = np.asarray(b)

    nrm_squared = np.sum(np.square(theta), axis=-1)
    linear_term = np.sum(theta * x, axis=-1) + b
    inner_prox_eta = eta * nrm_squared

    return x + theta * ((inner_prox(linear_term, inner_prox_eta) - linear_term) / nrm_squared)[..., np.newaxis]


def exponential_prox(x: ArrayLike, eta: ArrayLike | float):
    r"""
    Computes the proximal operator of :math:`f(z) = \exp(z)`

    :param x: The point(s) at which to compute the proximal operator.
    :param eta: The step-size(s) to use when computing the proximal operator.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    return x - wrightomega(x + np.log(eta))


def polylog_exp_prox(x: ArrayLike, eta: ArrayLike | float):
    r"""
    Computes the proximal operator of :math:`f(z) = -\operatorname{Li}_2(-\exp(z))`. This
    is exactly the integral of :math:`f'(x)=\ln(1+\exp(x))`.

    The resulting proximal operator satisfies the equation:
    ..math::
        x^* + \eta \ln(1 + \exp(x^*)) = x

    :param x: The point(s) at which to compute the proximal operator.
    :param eta: The step-size(s) to use when computing the proximal operator. Must be positive.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    t = x
    for _ in range(12): # 12 Newton iterations - seems to be enough
        x = x - (x + eta * np.logaddexp(0, x) - t) / (1 + eta * expit(x))
    return x


def square_pos_prox(x:ArrayLike, eta: ArrayLike | float):
    r"""
    Computes the proximal operator of :math:`f(z) = z^2 / 2` with :math:`\operatorname{dom}(f) = \mathbb{R}^+`.

    :param x: The point(s) at which to compute the proximal operator.
    :param eta: The step-size(s) to use when computing the proximal operator.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    return np.maximum(0, x / (eta + 1))


def square_prox(x: ArrayLike, eta: ArrayLike | float):
    r"""
    Computes the proximal operator of :math:`f(z) = z^2 / 2`

    :param x: The point(s) at which to compute the proximal operator.
    :param eta: The step-size(s) to use when computing the proximal operator.
    :return: The computed :math:`\operatorname{prox}_{\eta f}(x)`
    """
    return x / (1 + eta)


def compose_linear_perturb_reg(scalar_prox: Callable[[ArrayLike, ArrayLike | float], ArrayLike]):
    def prox_op(
            x: ArrayLike,
            eta: Union[ArrayLike, float],
            theta: ArrayLike,
            phi: ArrayLike,
            b: Union[ArrayLike, float],
            alpha: Union[ArrayLike, float]
    ):
        def linear_only_prox(u, step_size):
            return prox_linear_composed(u, step_size, scalar_prox, theta, b)
        return prox_regularized_perturbed(x, eta, linear_only_prox, phi, alpha)

    return prox_op