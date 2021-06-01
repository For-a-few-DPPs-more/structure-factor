#!/usr/bin/env python3
# coding=utf-8

import unittest
import numpy as np
from scipy.special import k0
from structure_factor.utils import (
    integrate_with_abs_odd_monomial,
    integrate_with_bessel_function_half_line,
)


class TestIntegrateWithAbsoluteValueOfOddMonomial(unittest.TestCase):
    # See Section 4 in Ogata
    def test_f1(self):
        # See Section 4 "Example 1" in Ogata
        f1 = lambda x: np.exp(-np.cosh(x)) / (1 + np.square(x))
        nu = 0
        actual = integrate_with_abs_odd_monomial(f1, nu)
        desired = 0.306354694925705
        np.testing.assert_almost_equal(actual, desired)

    def test_f2(self):
        # See Section 4 "Example 2" in Ogata
        f2 = lambda x: np.exp(-np.square(x))
        nu = 0
        actual = integrate_with_abs_odd_monomial(f2, nu)
        desired = 1.0
        np.testing.assert_almost_equal(actual, desired)


class TestIntegrateWithBesselFunctionHalfLine(unittest.TestCase):
    # See Section 5 paragraph "Numerical Examples" in Ogata
    def test_f3(self):
        f3 = lambda x: np.ones_like(x)
        nu = 0
        actual = integrate_with_bessel_function_half_line(f3, nu)
        desired = 1.0
        np.testing.assert_almost_equal(actual, desired)

    def test_f4(self):
        f4 = lambda x: x / (1 + np.square(x))
        nu = 0
        actual = integrate_with_bessel_function_half_line(f4, nu)
        desired = k0(1)
        np.testing.assert_almost_equal(actual, desired)
