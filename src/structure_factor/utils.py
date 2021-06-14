#!/usr/bin/env python3
# coding=utf-8

import numpy as np

from mpmath import fp as mpm
from scipy.special import j0, j1, jv, jn_zeros, y0, y1, yv
from scipy import interpolate

# todo bien renomer les variables


def binning_function(r_vector, data, bins_number):
    r"""
    binning function for ploting the scattering intensity
    """
    step = (np.max(r_vector) - np.min(r_vector)) / bins_number
    binned_data = []
    binned_r = []
    for i in range(1, bins_number + 1):
        index = (r_vector <= np.min(r_vector) + i * step) & (
            r_vector >= np.min(r_vector + (i - 1) * step)
        )
        binned_r.append(np.mean(r_vector[index]))
        binned_data.append(np.mean(data[index]))
    print(np.min(r_vector) + i * step, np.max(r_vector))
    return (binned_r, binned_data)


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


def bessel1(order, x):
    if order == 0:
        return j0(x)
    if order == 1:
        return j1(x)
    return jv(order, x)


def bessel1_zeros(order, nb_zeros):
    return jn_zeros(order, nb_zeros)


def bessel2(order, x):
    if order == 0:
        return y0(x)
    if order == 1:
        return y1(x)
    return yv(order, x)


def roots(d, N):
    # first N Roots of the Bessel J_(d/2-1) functions divided by pi.
    return np.array([mpm.besseljzero(d / 2 - 1, i + 1) for i in range(N)]) / np.pi


def get_x(h, zeros):
    return np.pi * psi(h * zeros) / h


def weight(d, zeros):
    return bessel2(d / 2 - 1, np.pi * zeros) / bessel1(d / 2, np.pi * zeros)


def psi(t):
    # equation 5.1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    return t * np.tanh((0.5 * np.pi) * np.sinh(t))


def d_psi(t):
    # equation 5.1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    threshold = 3.5  # threshold outside of which psi' plateaus to -1, 1
    out = np.sign(t)
    mask = np.abs(t) < threshold
    x = t[mask]
    out[mask] = np.pi * x * np.cosh(x) + np.sinh(np.pi * np.sinh(x))
    out[mask] /= 1.0 + np.cosh(np.pi * np.sinh(x))
    return out


def integrate_with_abs_odd_monomial(f, nu=0, h=0.1, n=100, f_even=False):
    # Section 1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    x = bessel1_zeros(nu, n)
    weights = bessel2(nu, x) / bessel1(nu + 1, x)  # equation 1.2
    x *= h / np.pi  # equivalent of xi variable
    # equation 1.1
    if f_even:
        return 2.0 * h * np.sum(weights * np.power(x, 2 * nu + 1) * f(x))
    return h * np.sum(weights * np.power(x, 2 * nu + 1) * (f(x) + f(-x)))


def integrate_with_bessel_function_half_line(f, nu=0, h=0.01, n=1000):
    # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    t = bessel1_zeros(nu, n)
    weights = bessel2(nu, t) / bessel1(nu + 1, t)  # equation 1.2
    t *= h / np.pi  # equivalent of xi variable
    # Change of variable equation 5.2
    x = (np.pi / h) * psi(t)
    return np.pi * np.sum(weights * f(x) * bessel1(nu, x) * d_psi(t))


def compute_scattering_intensity(k, data):
    X = data
    n = X.shape[0]
    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, X.T)), axis=1)))
    si /= n
    return si


class SymmetricFourierTransform:
    # todo renommer utils pars Ogata... car tout le mode est pour symmetric...  mettre le lien du papier Ogata, explicité les formule derier les fonction et variable, ajouter un test pour verifier que c'est bon
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
        kernel = bessel1(d / 2 - 1, self.x)  # J_(d/2-1)(pi*psi(h*ksi))
        w = weight(d, self._zeros)  # (Y_0(pi*zeros)/J_1(pi*zeros))
        self.dpsi = d_psi(h * self._zeros)  # dpsi(h*ksi)
        # pi*w*J_(d/2-1)(x)*dpsi(h*zeros)
        self._factor = np.pi * w * kernel * self.dpsi

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
        ret = (2 * np.pi) ** (self.d / 2) * np.sum(summation, axis=-1) / k ** self.d

        return ret, self.k_min
