import unittest

import numpy as np
from scipy.special import k0

from structure_factor.transforms import HankelTransformBaddourChouinard


def modified_gaussian(x, a, n):
    return np.exp(-np.square(a * x)) * np.power(x, n)


def ht_modified_gaussian(n, x, a):
    ht = np.exp(-np.square(x / (2 * a))) * np.power(x, n)
    ht *= (0.5 / a ** 2) ** (n + 1)
    return ht


def sinc(x, a=1.0):
    return np.sinc(x * (a / np.pi))


def ht_sinc(n, x, a):
    ht = np.full(x.shape, 1.0 / a ** 2)

    mask = x < a
    ht[mask] *= np.cos(np.pi * n / 2) * np.power(x[mask] / a, n)
    ht[mask] /= np.sqrt(1.0 - np.square(x[mask] / a))
    ht[mask] /= np.power(1.0 + np.sqrt(1.0 - np.square(x[mask] / a)), n)

    np.invert(mask, out=mask)  # x > a
    ht[mask] *= np.sin(n * np.arcsin(a / x[mask]))
    ht[mask] /= np.sqrt(np.square(x[mask] / a) - 1.0)
    return ht


def sinc_modified(x, a=1.0):
    return np.sinc(x * (a / np.pi)) / np.square(x)


def ht_sinc_modified(n, x, a):
    ht = np.full(x.shape, 1.0 / n)

    mask = x < a
    ht[mask] *= np.sin(np.pi * n / 2) * np.power(x[mask] / a, n)
    ht[mask] /= np.power(1.0 + np.sqrt(1.0 - np.square(x[mask] / a)), n)

    np.invert(mask, out=mask)  # x > a
    ht[mask] *= np.sin(n * np.arcsin(a / x[mask]))
    return ht


class TestHankelTransformWithBaddourChouinard(unittest.TestCase):

    order = 0
    ht = HankelTransformBaddourChouinard(order)

    def test_f1(self):
        a = 5
        f = lambda r: modified_gaussian(r, a, self.order)
        ht_f = lambda k: ht_modified_gaussian(self.order, k, a)

        r_max = 2
        nb_points = 64
        self.ht.compute_transformation_parameters(r_max, nb_points)
        k, actual = self.ht.transform(f)
        desired = ht_f(k)
        np.testing.assert_almost_equal(actual, desired)

    def test_f2(self):
        a = 5
        f = lambda r: sinc(r, a)
        ht_f = lambda k: ht_sinc(self.order, k, a)

        r_max = 30
        nb_points = 256
        self.ht.compute_transformation_parameters(r_max, nb_points)
        k, actual = self.ht.transform(f)
        desired = ht_f(k)
        np.testing.assert_almost_equal(
            actual,
            desired,
            decimal=0,
            err_msg="No worries if test fails, approximation wiggles around true transform (Gibbs phenomenon)",
        )
