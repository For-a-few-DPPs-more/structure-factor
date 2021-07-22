#!/usr/bin/env python3
# coding=utf-8

import unittest
import numpy as np
from scipy.special import k0

from hypton.transforms import (
    ogata_integrate_with_abs_monomial,
    HankelTransformOgata,
)


class TestOgataIntegrationAgainstAbsoluteValueOddMonomial(unittest.TestCase):
    # See Section 4 in Ogata
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
    ht.compute_transformation_parameters(0.01, 300)

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
