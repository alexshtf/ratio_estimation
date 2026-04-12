import numpy as np

from ratio_estimation.proximal import (
    exponential_prox,
    positive_square_prox,
    softplus_integral_prox,
    square_prox,
)


def test_square_prox_matches_closed_form() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    step_size = 0.5
    np.testing.assert_allclose(square_prox(x, step_size), x / (1.0 + step_size))


def test_positive_square_prox_clips_negative_values() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    step_size = 2.0
    np.testing.assert_allclose(positive_square_prox(x, step_size), [0.0, 0.0, 1.0])


def test_exponential_prox_satisfies_optimality_equation() -> None:
    x = np.array([-0.5, 0.0, 1.5])
    step_size = 0.7
    solution = exponential_prox(x, step_size)
    residual = solution + step_size * np.exp(solution) - x
    assert np.allclose(residual, 0.0, atol=1e-10)


def test_softplus_integral_prox_satisfies_optimality_equation() -> None:
    x = np.array([-1.0, 0.2, 2.0])
    step_size = 0.4
    solution = softplus_integral_prox(x, step_size)
    residual = solution + step_size * np.logaddexp(0.0, solution) - x
    assert np.allclose(residual, 0.0, atol=1e-10)
