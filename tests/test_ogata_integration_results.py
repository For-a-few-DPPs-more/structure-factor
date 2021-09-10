#!/usr/bin/env python3
# coding=utf-8

import unittest
import numpy as np
from scipy.special import k0

from hypton.transforms import HankelTransformOgata
from hypton.utils import bessel1, bessel1_zeros, bessel2


def ogata_integrate_with_abs_monomial(f, nu=0, h=0.1, n=100, f_even=False):
    # Section 1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    x = bessel1_zeros(nu, n)
    weights = bessel2(nu, x) / bessel1(nu + 1, x)  # equation 1.2
    x *= h / np.pi  # equivalent of xi variable
    # equation 1.1
    deg = 2 * nu + 1
    if f_even:
        return 2.0 * h * np.sum(weights * np.power(x, deg) * f(x), axis=-1)
    return h * np.sum(weights * np.power(x, deg) * (f(x) + f(-x)), axis=-1)


# def integrate_with_bessel_function_half_line(f, n=0, h=0.01, N=1000):
#     # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
#     assert n == np.floor(n)
#     t = bessel1_zeros(n, N)
#     weights = bessel2(n, t) / bessel1(n + 1, t)  # equation 1.2
#     t *= h / np.pi  # equivalent of xi variable
#     x = (np.pi / h) * psi(t)  # Change of variable equation 5.2
#     out = np.pi * np.sum(weights * f(x) * bessel1(n, x) * d_psi(t), axis=-1)
#     return out


# def hankel_transform_ogata(f, n=0, h=0.01, N=1000):
#     # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
#     assert n == np.floor(n)
#     t = bessel1_zeros(n, N)
#     weights = bessel2(n, t) / bessel1(n + 1, t)  # equation 1.2
#     t *= h / np.pi  # equivalent of xi variable
#     x = (np.pi / h) * psi(t)  # Change of variable equation 5.2
#     out = np.pi * np.sum(weights * f(x) * bessel1(n, x) * d_psi(t), axis=-1)
#     return out


class TestOgataIntegrationAgainstAbsoluteValueOddMonomial(unittest.TestCase):
    # See Section 4 in Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    def test_example1_section4(self):
        # See Section 4 "Example 1" in Ogata
        f = lambda x: np.exp(-np.cosh(x)) / (1 + np.square(x))
        order = 0
        actual = ogata_integrate_with_abs_monomial(f, order)
        desired = 0.306354694925705
        np.testing.assert_almost_equal(actual, desired)

    def test_example2_section4(self):
        # See Section 4 "Example 2" in Ogata
        f = lambda x: np.exp(-np.square(x))
        order = 0
        actual = ogata_integrate_with_abs_monomial(f, order)
        desired = 1.0
        np.testing.assert_almost_equal(actual, desired)


class TestOgataHankelTransformEvaluatedAtOne(unittest.TestCase):
    # See Section 5 paragraph "Numerical Examples" in Ogata
    order = 0
    ht = HankelTransformOgata(order)
    ht.compute_transformation_parameters(step_size=0.01, nb_points=300)

    def test_numerical_example1_section5(self):
        f = lambda r: 1.0 / r
        _, actual = self.ht.transform(f, 1)
        desired = 1.0
        np.testing.assert_almost_equal(actual, desired)

    def test_numerical_example2_section5(self):
        f = lambda r: 1.0 / (1.0 + r ** 2)
        _, actual = self.ht.transform(f, 1)
        desired = k0(1)
        np.testing.assert_almost_equal(actual, desired)
