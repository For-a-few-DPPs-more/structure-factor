#!/usr/bin/env python3
# coding=utf-8

import numpy as np
from scipy import interpolate

from hypton.utils import bessel1, bessel2, bessel1_zeros


class RadiallySymmetricFourierTransform:
    """Compute the Fourier transform of a radially symmetric function using the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_."""

    def __init__(self, dimension=0):
        assert isinstance(dimension, int)
        assert dimension % 2 == 0
        self.d = dimension
        self.rmax = None

    def transform(self, f, k, method="Ogata", **params):
        """Evaluate the Fourier transform :math:`F[f](k)` of the radially symmetric function :math:`f` at :math:`k` using the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_.

        .. math::

            k^{d/2-1} F(k)
            = (2 \pi)^{d/2}
            \int_{0}^{+\infty}
                r^{d/2-1}
                f(r)
                J_{d/2-1}(kr)
                r
                \mathrm{d}r
            = (2 \pi)^{d/2}
            H_{d/2-1}[\cdot^{d/2-1} f(\cdot)](k)

        Args:
            f (callable): function to transform
            k (scalar or numpy.ndarray): point(s) where the Fourier tranform is to evaluated.
            method (str, optional): name of the method used to compute the underlying Hankel transform: ``"Ogata" or "BaddourChouinard"``. Defaults to ``"Ogata"``.

        Keyword Args:
            interpolation (dict): description
            key2 (int): description

        Returns:
            tuple: :math:`(k, F[f](k))`
        """
        d = self.d
        order = d // 2 - 1
        ht = self._get_hankel_transformer(order, method)
        interp_params = params.pop("interpolation", dict())
        ht.compute_transformation_parameters(**params)
        g = lambda r: f(r) * r ** order
        k, F_k = ht.transform(g, k, **interp_params)
        F_k *= (2 * np.pi) ** (d / 2)
        if order != 0:  # F_k /= k^(d/2-1)
            F_k /= k ** order
        return k, F_k

    def _get_hankel_transformer(self, order, method):
        hankel_transformer = {
            "Ogata": HankelTransformOgata,
            "BaddourChouinard": HankelTransformBaddourChouinard,
        }
        return hankel_transformer[method]

    def compute_k_min(self, step_size):
        # todo add docstring
        rmax = self.rmax
        # todo explain the hard coded choice of 2.7
        return (2.7 * np.pi) / (rmax * step_size)


class HankelTransform(object):
    r"""`Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform>`_

    .. seealso::

        - :py:class:`RadiallySymmetricFourierTransform`,
        - :py:class:`HankelTransformBaddourChouinard`,
        - :py:class:`HankelTransformOgata`.
    """

    def __init__(self, order):
        assert order == np.floor(order)
        self.order = order


class HankelTransformBaddourChouinard(HankelTransform):
    r"""Computation of the forward `Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform>`_

    .. math::

        H_\nu[f](k) = \int_0^\infty f(r) J_\nu(kr) r \mathrm{d}r,

    using the method of :cite:`BaCh15` considering that the input function is space-limited, i.e., :math:`f(r)=0` for :math:`r>r_{max}`.

    .. seealso::

        - MatLab code of Baddour Chouinard https://openresearchsoftware.metajnl.com/articles/10.5334/jors.82/,
        - `Pyhank Python package <https://pypi.org/project/pyhank/>`_.
    """

    def __init__(self, order=0):
        super(HankelTransformBaddourChouinard, self).__init__(order=order)
        self.bessel_zeros = None
        self.rmax = None  # R in :cite:`BaCh15` Section 4.B
        self.transformation_matrix = None  # Y in :cite:`BaCh15` Section 6.A

    def compute_transformation_parameters(self, rmax, nb_points):
        """Compute parameters involved in the computation of the corresponding Hankel-type transform using the discretization scheme of :cite:`BaCh15`.

        The following attributes are defined
        - :py:attr:`HankelTransformBaddourChouinard.bessel_zeros`
        - :py:attr:`HankelTransformBaddourChouinard.r_max`
        - :py:attr:`HankelTransformBaddourChouinard.transformation_matrix`

        Args:
            step_size (float, optional): Step size of the discretization scheme. Defaults to 0.01.
            nb_points (int, optional): Number of quadrature nodes. Defaults to 300.
        """
        n = self.order
        bessel_zeros = bessel1_zeros(n, nb_points)
        jk, jN = bessel_zeros[:-1], bessel_zeros[-1]
        # Section 6.A Transformation matrix
        Y = bessel1(n, np.outer(jk / jN, jk)) / np.square(bessel1(n + 1, jk))
        Y *= 2 / jN

        self.bessel_zeros = bessel_zeros
        self.rmax = rmax
        self.transformation_matrix = Y

    def transform(self, f, k=None, **interpolation_params):
        r"""Compute the Hankel transform

        .. math::

            H_\nu[f](k) = \int_0^\infty f(r) J_\nu(kr) r \mathrm{d}r,

        Args:
            f (callable): function to be Hankel transformed
            k (np.ndarray, optional): points of evaluation of the Hankel transform. Defaults to None.

            - If ``k`` is None (default), ``k = self.bessel_zeros[:-1] / self.rmax`` derived from :py:meth:`HankelTransformBaddourChouinard.compute_transformation_parameters`.
            - If ``k`` is provided, the Hankel transform is first computed at the above k values (case k is None), then interpolated using :py:func:`scipy.interpolate.interp1d` with ``interpolation_params`` and finally evaluated at the provided ``k`` values.

        Keyword Args:

            - same keyword arguments as :py:func:`scipy.interpolate.interp1d`: ``"fill_value"``, ``"kind"``, etc.

        Returns:
            tuple (scalar or np.ndarray, scalar or np.ndarray): :math:`k, H[f](k)`
        """
        assert callable(f)
        rmax = self.rmax
        Y = self.transformation_matrix
        jk, jN = self.bessel_zeros[:-1], self.bessel_zeros[-1]
        r = jk * (rmax / jN)
        ht_k = (rmax ** 2 / jN) * Y.dot(f(r))  # Equation (23)
        _k = jk / rmax
        if k is not None:
            interpolation_params["assume_sorted"] = True
            interpolation_params.setdefault("fill_value", "extrapolate")
            interpolation_params.setdefault("kind", "cubic")
            ht = interpolate.interp1d(_k, ht_k, **interpolation_params)
            return k, ht(k)
        return _k, ht_k


class HankelTransformOgata(HankelTransform):
    r"""Computation of the forward `Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform>`_

    .. math::

        H_\nu(k) = \int_0^\infty f(r) J_\nu(kr) r \mathrm{d}r,

    using the method of :cite:`Oga05` Section 5.

    .. seealso::

        - `Hankel Python package <https://joss.theoj.org/papers/10.21105/joss.01397>`_
    """

    def __init__(self, order=0):
        super(HankelTransformOgata, self).__init__(order=order)
        self.nodes, self.weights = None, None

    def compute_transformation_parameters(
        self, rmax=None, nb_points=300, step_size=0.01
    ):
        """Compute the quadrature nodes and weights used by :cite:`Oga05` Equation (5.2) to evaluate the corresponding Hankel-type transform.

        Args:
            step_size (float, optional): Step size of the discretization scheme. Defaults to 0.01.
            nb_points (int, optional): Number of quadrature nodes. Defaults to 300.

        Returns:
            tuple (np.ndarray, np.ndarray): quadrature nodes and weights
        """
        n = self.order
        h = step_size
        N = nb_points
        #! self.rmax is not defined
        self.rmax = rmax
        t = bessel1_zeros(n, N)
        weights = bessel2(n, t) / bessel1(n + 1, t)  # Equation (1.2)
        t *= h / np.pi  # Equivalent of xi variable
        weights *= self._d_psi(t)
        nodes = (np.pi / h) * self._psi(t)  # Change of variable Equation (5.1)
        self.nodes, self.weights = nodes, weights
        return nodes, weights

    def transform(self, f, k):
        r"""Compute Hankel transform :math:`H[f](k)` of ``f`` evaluated at ``k``, following the work of :cite:`Oga05` Section 5.
        Please call :py:meth:`HankelTransformOgata.compute_transformation_parameters` before :py:meth:`HankelTransformOgata.compute_transform`.

        Args:
            f (callable): function to be Hankel transformed
            k (np.ndarray, optional): points of evaluation of the Hankel transform. Defaults to None.

        Returns:
            tuple(np.ndarray): :math:`k, H[f](k)`.
        """
        assert callable(f)
        n = self.order
        w = self.weights
        x = self.nodes
        k_ = k[:, None] if isinstance(k, np.ndarray) else k
        g = lambda r: f(r / k_) * r  # or f(r / k_) * (r / k**2)
        H_k = np.pi * np.sum(w * g(x) * bessel1(n, x), axis=-1)
        H_k /= k ** 2
        return k, H_k

    @staticmethod
    def _psi(t):
        """Function involved in the change of variable used by :cite:`Oga05` Equation (5.1)"""
        return t * np.tanh((0.5 * np.pi) * np.sinh(t))

    @staticmethod
    def _d_psi(t):
        """Function involved in the change of variable used by :cite:`Oga05` Equation (5.1)"""
        threshold = 3.5  # threshold outside of which psi' plateaus to -1, 1
        out = np.sign(t)
        mask = np.abs(t) < threshold
        x = t[mask]
        out[mask] = np.pi * x * np.cosh(x) + np.sinh(np.pi * np.sinh(x))
        out[mask] /= 1.0 + np.cosh(np.pi * np.sinh(x))
        return out
