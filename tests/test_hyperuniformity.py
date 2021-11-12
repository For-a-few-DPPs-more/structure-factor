import numpy as np
import pytest

from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.utils import structure_factor_ginibre, structure_factor_poisson

k = np.linspace(0, 10, 100)


def f(c, alpha, x):
    return c * x ** alpha


x_1 = np.linspace(0, 3, 100)
x_2 = np.linspace(0.5, 2, 50)


@pytest.mark.parametrize(
    "k, sf_k, expected",
    [
        (k, structure_factor_poisson(k), False),
        (k, structure_factor_ginibre(k), True),
    ],
)
def test_effective_hyperuniformity(k, sf_k, expected):
    # verify that the hyperuniformity index for the ginibre ensemble is less than 1e-3
    hyperuniformity_test = Hyperuniformity(k, sf_k)
    index_H, _ = hyperuniformity_test.effective_hyperuniformity(k_norm_stop=4)
    result = index_H < 1e-3
    assert result == expected


@pytest.mark.parametrize(
    "x, fx, c, alpha",
    [
        (x_1, f(8, 2, x_1), 8, 2),
        (x_2, f(6, 0.5, x_2), 6, 0.5),
    ],
)
def test_hyperuniformity_class(x, fx, c, alpha):
    test = Hyperuniformity(x, fx)
    assert alpha, c == test.hyperuniformity_class()
