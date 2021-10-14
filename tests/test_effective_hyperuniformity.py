import numpy as np
import pytest

from hypton.effective_hyperuniform import EffectiveHyperuniformity
from hypton.utils import structure_factor_ginibre, structure_factor_poisson

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
# ? I don't understand what is tested: the value of the hyperuniformity index or the fact that is it small (< 1e-3)
# todo use a more explicit name, test_small_hyperuniformity_index or test_hyperuniformity_index_should_be...
def test_hyperuniformity_index(k, sf_k, expected):
    # todo rename hyp_test, it reads like hypothesis test
    hyp_test = EffectiveHyperuniformity(k, sf_k)
    index_H, _ = hyp_test.index_H(norm_k_stop=4)
    result = index_H < 1e-3
    assert result == expected


@pytest.mark.parametrize(
    "x, fx, c, alpha",
    [
        (x_1, f(8, 2, x_1), 8, 2),
        (x_2, f(6, 0.5, x_2), 6, 0.5),
    ],
)
def test_power_decay(x, fx, c, alpha):
    test = EffectiveHyperuniformity(x, fx)
    assert alpha, c == test.power_decay()
