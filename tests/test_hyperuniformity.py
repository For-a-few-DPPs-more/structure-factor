import numpy as np
import pytest

from structure_factor.hyperuniformity import (
    effective_hyperuniformity,
    hyperuniformity_class,
    multiscale_test,
)
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
    summary = effective_hyperuniformity(k, sf_k, k_norm_stop=4)
    result = summary["H"] < 1e-3
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
    result = hyperuniformity_class(x, fx)
    assert alpha == result["alpha"]
    assert c == result["c"]


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
    result = hyperuniformity_class(k, sf_k, k_norm_stop=0.001)
    diff_alpha = result["alpha"] - expected_alpha
    np.testing.assert_almost_equal(diff_alpha, 0, decimal=3)
