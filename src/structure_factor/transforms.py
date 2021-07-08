#!/usr/bin/env python3
# coding=utf-8

import numpy as np

from structure_factor.utils import bessel1, bessel2, bessel1_zeros
from scipy import interpolate


class RadiallySymmetricFourierTransform:
    """Compute the Fourier transform of a radially symmetric function  using the `correspondance with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_"""

    def __init__(self, dimension=0):
        assert isinstance(dimension, int)
        assert dimension % 2 == 0
        self.d = dimension
        self._hankel_transform_methods = ("Ogata", "BaddourChouinard")

    def transform(self, f, k, method="Ogata", **params):
        d = self.d
        order = d // 2 - 1
        ht = self._get_hankel_transformer(order, method)
        interp_params = params.pop("interpolation", dict())
        ht.compute_transformation_parameters(**params)
        g = lambda r: f(r) * r ** order
        F_k = (2 * np.pi) ** (d / 2) * ht.transform(g, k, **interp_params)
        if order != 0:  # F_k /= k^(d/2-1)
            k_ = k[:, None] if isinstance(k, np.ndarray) else k
            np.power(k_, order, out=k_)
            F_k /= k_
        return F_k

    @staticmethod
    def _get_hankel_transformer(self, order, method):
        assert method in self._hankel_transform_methods
        if method == "Ogata":
            return HankelTransformOgata(order=order)
        if method == "BaddourChouinard":
            return HankelTransformBaddourChouinard(order=order)


class HankelTransform(object):
    def __init__(self, order):
        assert order == np.floor(order)
        self.order = order


class HankelTransformBaddourChouinard(HankelTransform):
    """Computation of the forward Hankel transform using the method of :cite:`BaCh15` considering that the input function is space-limited, i.e., :math:`f(r)=0` for :math:`r>r_{max}`.

    .. seealso::

        - MatLab code of Baddour Chouinard https://openresearchsoftware.metajnl.com/articles/10.5334/jors.82/
        - pyhank https://pypi.org/project/pyhank/
    """

    def __init__(self, order=0):
        super(HankelTransformBaddourChouinard, self).__init__(order=order)
        self.bessel_zeros = None
        self.r_max = None  # R in Section 4.B Space-Limited Function
        self.transformation_matrix = None  # Y in Section 6.A

    def compute_transformation_parameters(self, r_max, nb_points):
        n = self.order
        bessel_zeros = bessel1_zeros(n, nb_points)
        jk, jN = bessel_zeros[:-1], bessel_zeros[-1]
        # Section 6.A Transformation matrix
        Y = bessel1(n, np.outer(jk / jN, jk)) / np.square(bessel1(n + 1, jk))
        Y *= 2 / jN

        self.bessel_zeros = bessel_zeros
        self.r_max = r_max
        self.transformation_matrix = Y

    def transform(self, f, k=None, **interpolation_params):
        """Compute Hankel transform :math:`HT[f](k)` of ``f`` evaluated at ``k``.
        If ``k`` is None, values considered are ``k = self.bessel_zeros[:-1] / self.r_max`` derived from :py:meth:`HankelTransformBaddourChouinard.compute_transformation_parameters`.
        If ``k`` is provided, the Hankel transform is first computed for the above k values (case k is None), then interpolated using :py:func:`scipy.interpolate.interp1d` with ``interpolation_params`` and finally evaluated at the provided ``k`` values.

        Args:
            f (callable): function to be Hankel transformed
            k (np.ndarray, optional): points of evaluation of the Hankel transform. Defaults to None.

        Returns:
            tuple(np.ndarray): :math:`k, HT[f](k)`
        """
        assert callable(f)
        r_max = self.r_max
        Y = self.transformation_matrix
        jk, jN = self.bessel_zeros[:-1], self.bessel_zeros[-1]
        r = jk * (r_max / jN)
        ht_k = (r_max ** 2 / jN) * Y.dot(f(r))  # Equation (23)
        _k = jk / r_max
        if k is not None:
            interpolation_params["assume_sorted"] = True
            ht = interpolate.interp1d(_k, ht_k, **interpolation_params)
            return k, ht(k)
        return _k, ht_k


class HankelTransformOgata(HankelTransform):
    """Computation of the forward Hankel transform using the method of :cite:`Oga05` Section 5.

    .. seealso::

        hankel https://joss.theoj.org/papers/10.21105/joss.01397
    """

    def __init__(self, order=0):
        super(HankelTransformOgata, self).__init__(order=order)
        self.nodes, self.weights = None, None

    def compute_transformation_parameters(self, step_size=0.01, nb_points=300):
        """Compute the quadrature nodes and weights used by :cite:`Oga05` Equation (5.2) to evaluate the Hankel-type transform.

        Args:
            step_size (float, optional): Step size of the discretization scheme. Defaults to 0.01.
            nb_points (int, optional): Number of quadrature nodes. Defaults to 300.

        Returns:
            tuple(np.ndarray): quadrature nodes and weights
        """
        n = self.order
        h = step_size
        N = nb_points

        t = bessel1_zeros(n, N)
        weights = bessel2(n, t) / bessel1(n + 1, t)  # Equation (1.2)
        t *= h / np.pi  # Equivalent of xi variable
        weights *= self.d_psi(t)
        nodes = (np.pi / h) * self.psi(t)  # Change of variable Equation (5.1)
        self.nodes, self.weights = nodes, weights
        return nodes, weights

    def transform(self, f, k):
        """Compute Hankel transform :math:`HT[f](k)` of ``f`` evaluated at ``k``, following the work of :cite:`Oga05` Section 5.

        Args:
            f (callable): function to be Hankel transformed
            k (np.ndarray, optional): points of evaluation of the Hankel transform. Defaults to None.

        Returns:
            tuple(np.ndarray): :math:`k, HT[f](k)`
        """
        assert callable(f)
        n = self.order
        w = self.weights
        x = self.nodes
        k_ = k[:, None] if isinstance(k, np.ndarray) else k
        g = lambda r: f(r / k_) * r  # or f(r / k_) * (r / k**2)
        H_k = np.pi * np.sum(w * g(x) * bessel1(n, x), axis=-1)
        # H_k /= k^2
        np.square(k_, out=k_)
        H_k /= k_
        return k, H_k

    @staticmethod
    def psi(t):
        """Function involved in the change of variable used by :cite:`Oga05` Equation (5.1)"""
        return t * np.tanh((0.5 * np.pi) * np.sinh(t))

    @staticmethod
    def d_psi(t):
        """Function involved in the change of variable used by :cite:`Oga05` Equation (5.1)"""
        threshold = 3.5  # threshold outside of which psi' plateaus to -1, 1
        out = np.sign(t)
        mask = np.abs(t) < threshold
        x = t[mask]
        out[mask] = np.pi * x * np.cosh(x) + np.sinh(np.pi * np.sinh(x))
        out[mask] /= 1.0 + np.cosh(np.pi * np.sinh(x))
        return out


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


def integrate_with_bessel_function_half_line(f, n=0, h=0.01, N=1000):
    # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    assert n == np.floor(n)
    t = bessel1_zeros(n, N)
    weights = bessel2(n, t) / bessel1(n + 1, t)  # equation 1.2
    t *= h / np.pi  # equivalent of xi variable
    x = (np.pi / h) * psi(t)  # Change of variable equation 5.2
    out = np.pi * np.sum(weights * f(x) * bessel1(n, x) * d_psi(t), axis=-1)
    return out


def hankel_transform_ogata(f, n=0, h=0.01, N=1000):
    # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    assert n == np.floor(n)
    t = bessel1_zeros(n, N)
    weights = bessel2(n, t) / bessel1(n + 1, t)  # equation 1.2
    t *= h / np.pi  # equivalent of xi variable
    x = (np.pi / h) * psi(t)  # Change of variable equation 5.2
    out = np.pi * np.sum(weights * f(x) * bessel1(n, x) * d_psi(t), axis=-1)
    return out


def ht_baddour_chouinard(function, order, r_max, nb_points, mode="Y"):
    n = order
    bessel1_zeros = bessel1_zeros(n, nb_points)
    jk, jN = bessel1_zeros[:-1], bessel1_zeros[-1]
    r = jk * (r_max / jN)
    k = jk / r_max

    if mode == "Y":  # for space limited function
        H = bessel1(n, np.outer(jk / jN, jk)) / np.square(bessel1(n + 1, jk))
        H *= 2 / jN
        ht_k = H.dot(function(r))
        ht_k *= r_max ** 2 / jN
        return k, ht_k

    elif mode == "T":  # for band limited function
        H = bessel1(n, np.outer(jk / jN, jk))
        Jn1 = bessel1(n + 1, jk)
        H /= np.outer(Jn1, Jn1)
        H *= 2 / jN
        ht_k = H.dot(function(r) / Jn1)
        ht_k *= r_max ** 2 / jN
        return k, ht_k
