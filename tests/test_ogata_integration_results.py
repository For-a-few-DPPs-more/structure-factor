import numpy as np
import pytest
from scipy.special import k0

from structure_factor.transforms import HankelTransformOgata
from structure_factor.utils import bessel1, bessel1_zeros, bessel2


def integrate_against_abs_monomial(f, nu=0, h=0.1, n=100, f_even=False):
    # Section 1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    x = bessel1_zeros(nu, n)
    weights = bessel2(nu, x) / bessel1(nu + 1, x)  # equation 1.2
    x *= h / np.pi  # equivalent of xi variable
    # equation 1.1
    deg = 2 * nu + 1
    if f_even:
        return 2.0 * h * np.sum(weights * np.power(x, deg) * f(x), axis=-1)
    return h * np.sum(weights * np.power(x, deg) * (f(x) + f(-x)), axis=-1)


# See Section 4 in Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf


def test_section4_example1():
    f = lambda x: np.exp(-np.cosh(x)) / (1 + np.square(x))
    order = 0
    actual = integrate_against_abs_monomial(f, order)
    desired = 0.306354694925705
    np.testing.assert_almost_equal(actual, desired)


def test_section4_example2():
    f = lambda x: np.exp(-np.square(x))
    order = 0
    actual = integrate_against_abs_monomial(f, order)
    desired = 1.0
    np.testing.assert_almost_equal(actual, desired)


# See Section 5 paragraph "Numerical Examples" in Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf


@pytest.fixture
def ht():
    order = 0
    ht = HankelTransformOgata(order)
    ht.compute_transformation_parameters(step_size=0.01, nb_points=300)
    return ht


def test_section5_example1(ht):
    f = lambda r: 1.0 / r
    _, actual = ht.transform(f, 1)
    desired = 1.0
    np.testing.assert_almost_equal(actual, desired)


def test_section5_example2(ht):
    f = lambda r: 1.0 / (1.0 + r ** 2)
    _, actual = ht.transform(f, 1)
    desired = k0(1)
    np.testing.assert_almost_equal(actual, desired)
