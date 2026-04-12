"""Proximal operators used by the online ratio learners."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, wrightomega

FloatArray = NDArray[np.float64]
ScalarProx = Callable[[ArrayLike, float], ArrayLike | float]


def prox_with_regularizer_shift(
    x: ArrayLike,
    step_size: float,
    inner_prox: ScalarProx,
    shift: ArrayLike,
    regularization: float,
) -> FloatArray:
    """Apply the prox of h(x) + <shift, x> + (regularization / 2) ||x||²."""
    x_array = np.asarray(x, dtype=float)
    shift_array = np.asarray(shift, dtype=float)
    denominator = 1.0 + step_size * regularization
    inner_argument = (x_array - step_size * shift_array) / denominator
    inner_step_size = step_size / denominator
    return np.asarray(inner_prox(inner_argument, inner_step_size), dtype=float)


def prox_affine_composition(
    x: ArrayLike,
    step_size: float,
    scalar_prox: ScalarProx,
    direction: ArrayLike,
    bias: float = 0.0,
) -> FloatArray:
    """Apply the prox of g(<direction, x> + bias) through the scalar prox of g."""
    x_array = np.asarray(x, dtype=float)
    direction_array = np.asarray(direction, dtype=float)
    direction_norm_sq = float(np.dot(direction_array, direction_array))
    if direction_norm_sq == 0.0:
        return x_array
    linear_term = float(np.dot(direction_array, x_array) + bias)
    updated_linear_term = float(
        np.asarray(scalar_prox(linear_term, step_size * direction_norm_sq), dtype=float)
    )
    return x_array + direction_array * ((updated_linear_term - linear_term) / direction_norm_sq)


def exponential_prox(x: ArrayLike, step_size: float) -> FloatArray:
    """Apply the prox of z ↦ exp(z)."""
    x_array = np.asarray(x, dtype=float)
    prox_value = np.real_if_close(wrightomega(x_array + np.log(step_size)))
    return np.asarray(x_array - prox_value, dtype=float)


def softplus_integral_prox(x: ArrayLike, step_size: float) -> FloatArray:
    """Apply the prox of the integral of softplus."""
    target = np.asarray(x, dtype=float)
    solution = target.copy()
    for _ in range(12):
        residual = solution + step_size * np.logaddexp(0.0, solution) - target
        slope = 1.0 + step_size * expit(solution)
        solution = solution - residual / slope
    return solution


def positive_square_prox(x: ArrayLike, step_size: float) -> FloatArray:
    """Apply the prox of z ↦ ½ max(0, z)²."""
    x_array = np.asarray(x, dtype=float)
    return np.maximum(0.0, x_array / (1.0 + step_size))


def square_prox(x: ArrayLike, step_size: float) -> FloatArray:
    """Apply the prox of z ↦ ½ z²."""
    x_array = np.asarray(x, dtype=float)
    return x_array / (1.0 + step_size)
