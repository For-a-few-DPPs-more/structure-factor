#!/usr/bin/env python3
# coding=utf-8

import numpy as np
from mpmath import fp as mpm
from scipy.special import yv, jv
from scipy import interpolate


def get_random_number_generator(seed):
    """Turn seed into a np.random.Generator instance"""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(
        "seed must be None, an np.random.Generator or an integer (int, np.integer)"
    )
    return np.random.default_rng(seed)


def roots(d, N):
    # first N Roots of the Bessel J_(d/2-1) functions divided by pi.
    return np.array([mpm.besseljzero(d / 2 - 1, i + 1) for i in range(N)]) / np.pi


def psi(t):
    return t * np.tanh(np.pi * np.sinh(t) / 2)


def get_x(h, zeros):
    return np.pi * psi(h * zeros) / h


def weight(d, zeros):
    return yv(d / 2 - 1, np.pi * zeros) / jv(d / 2, np.pi * zeros)


def d_psi(t):
    t = np.array(t, dtype=float)
    d_psi = np.ones_like(t)
    exact_t = t < 6
    t = t[exact_t]
    d_psi[exact_t] = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (
        1.0 + np.cosh(np.pi * np.sinh(t))
    )
    return d_psi


def estimate_scattering_intensity(wave_vectors, points):
    scattering_intensity = (
        np.abs(np.sum(np.exp(-1j * np.dot(wave_vectors, points.T)), axis=1)) ** 2
    )
    scattering_intensity /= points.shape[0]
    return scattering_intensity


class SymmetricFourierTransform:
    """
    implement Symmetric Fourier transform based on OGATA paper "Integration Based On Bessel Function", with a change of variable allowing to
    approximate the Symmetric Fourier transform, needed to approximate the structure factor of a set of data, by first approximating the pair
    correlation function (of just having the exact function), and taking the Fourier transform of the total pair correlation function .
    self....
    """

    # todo give more explicit names to attributes: ex zeros -> quadrature_nodes
    def __init__(self, N, d=2, h=0.1):
        """
        Args:
            d (int): dimension of the space. Defaults to 2.
            N (int): number of sample points used to approximate the integral by a sum.
            h (float): step size in the sum. Defaults to 0.1.
            à ajouter les methods
        """
        if not isinstance(N, int):
            raise TypeError("N should be an integer.")
        self.N = N
        self.d = d
        self.step = h

        self.k_min = 0.0
        self._zeros = roots(d, N)  # Xi
        self.x = get_x(h, self._zeros)  # pi*psi(h*ksi/pi)/h
        self.kernel = jv(d / 2 - 1, self.x)  # J_(d/2-1)(pi*psi(h*ksi))
        self.w = weight(d, self._zeros)  # (Y_0(pi*zeros)/J_1(pi*zeros))
        self.dpsi = d_psi(h * self._zeros)  # dpsi(h*ksi)
        # pi*w*J_(d/2-1)(x)*dpsi(h*zeros)
        self._factor = np.pi * self.w * self.kernel * self.dpsi

    # todo rename function eg interpolate_correlation_function, interpolate
    def interpolate_correlation_function(self, r_vector, data_g):
        """given evaluations of the pair correlation function (g), it returns an interpolation of the total correlation function (h=g-1)

        Args:
            r_vector (np.array): vector containing the radius on which the pair correlation function is evaluated.
            data_g (np.array_like(r_vector)): vector containing the evaluations of the pair correlation function on r_vec.
        """

        return interpolate.interp1d(
            r_vector, data_g - 1.0, axis=0, fill_value="extrapolate", kind="cubic"
        )

    def _get_series(self, f, k, alpha):
        with np.errstate(divide="ignore"):  # numpy safely divides by 0
            args = np.divide.outer(self.x, k).T  # x/k
        # pi*w*J_(d/2-1)(x)*dpsi(h*zeros)f(x/k)J_(d/2-1)(x)*x**(d/2)
        return self._factor * (f(args) - 1 * alpha) * (self.x ** (self.d / 2))

    # todo give more explicit names to arguments k -> wave_lengths, g -> pcf (pair correlation function)
    def transform(
        self,
        k,
        g=None,
        r_vector=None,
        data_g=None,
    ):
        """Return an approximation of the symmetric Fourier transform of the total correlation function (h = g-1), and an estimation of the minimum confidence wave length.

        Args:
            k (np.array): vector containing the wavelength on which we want to approximate the structure factor.
            g (func): Pair correlation function if it's  known, else it will be approximated using data_g and r_vector. Defaults to None ( in this case r_vector and data_g should be provided).
            r_vector (np.array): vector containing the radius on which the pair correlation function is evaluated . Defaults to None.
            data_g (np.array_like(r_vector)): vector containing the evaluations of the pair correlation function on r_vec. Defaults to None.


        Returns:
            ret (np.array_like(k)): estimation of the fourier transform of the total correlation function.
            k_min (float): minimum confidence value of wavelength.
        """
        k = np.array(k)
        # todo naming is confusing between f, g and h = (g - 1)
        if g is None:
            f = self.interpolate_correlation_function(r_vector, data_g)
            self.k_min = (np.pi * 3.2) / (self.step * np.max(r_vector))
            summation = self._get_series(f, k, alpha=0)  # pi*w*J0(x)
        else:
            self.k_min = np.min(k)
            summation = self._get_series(g, k, alpha=1)  # pi*w*J0(x)

        # 2pi/k**2*sum(pi*w*f(x/k)J_0(x)*dpsi(h*ksi)*x)
        ret = (
            (2 * np.pi) ** (self.d / 2)
            * np.sum(summation, axis=-1)
            / np.array(k ** self.d)
        )

        return ret, self.k_min
