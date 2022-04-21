import numpy as np
import pytest

from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import (
    GinibrePointProcess,
    HomogeneousPoissonPointProcess,
)


@pytest.mark.parametrize(
    "sf, expected",
    [
        (HomogeneousPoissonPointProcess.structure_factor, False),
        (GinibrePointProcess.structure_factor, True),
    ],
)
def test_effective_hyperuniformity(sf, expected):
    # verify that the hyperuniformity index for the ginibre ensemble is less than 1e-3
    k = np.linspace(0, 10, 100)
    sf_k = sf(k)
    hyperuniformity_test = Hyperuniformity(k, sf_k)
    index_H, _ = hyperuniformity_test.effective_hyperuniformity(k_norm_stop=4)
    result = index_H < 1e-3
    assert result == expected


def f(c, alpha, x):
    return c * x ** alpha


x_1 = np.linspace(0, 3, 100)
x_2 = np.linspace(0.5, 2, 50)


@pytest.mark.parametrize(
    "x, fx, c, alpha",
    [
        (x_1, f(8, 2, x_1), 8, 2),
        (x_2, f(6, 0.5, x_2), 6, 0.5),
    ],
)
def test_hyperuniformity_class_on_polynomial(x, fx, c, alpha):
    test = Hyperuniformity(x, fx)
    assert alpha, c == test.hyperuniformity_class()


@pytest.mark.parametrize(
    "sf, expected_alpha",
    [
        (GinibrePointProcess.structure_factor, 2),
    ],
)
def test_hyperuniformity_class_ginibre(sf, expected_alpha):
    # verify that the hyperuniformity index for the ginibre ensemble is less than 1e-3
    k = np.linspace(0, 1, 3000)
    sf_k = sf(k)
    hyperuniformity_test = Hyperuniformity(k, sf_k)
    alpha, _ = hyperuniformity_test.hyperuniformity_class(k_norm_stop=0.001)
    diff_alpha = alpha - expected_alpha
    np.testing.assert_almost_equal(diff_alpha, 0, decimal=3)


@pytest.mark.parametrize(
    "proba_list, estimation_list, expected_result",
    [(1, 2, 2), ([1, 2, 3], [2, 4, 1], 2)],
)
def test_multiscale_test_on_simple_values(proba_list, estimation_list, expected_result):
    hyperuniformity_test = Hyperuniformity(sf_k_min_list=estimation_list)
    result = hyperuniformity_test.multiscale_test(proba_list)
    print(result, expected_result)
    np.testing.assert_almost_equal(result, expected_result)
