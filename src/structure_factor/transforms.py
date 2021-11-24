#!/usr/bin/env python3
# coding=utf-8

import numpy as np
from scipy import interpolate

import structure_factor.utils as utils
from structure_factor.utils import bessel1, bessel1_zeros, bessel2


class RadiallySymmetricFourierTransform:
    r"""Compute the Fourier transform of a radially symmetric function using the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_.

    Args:
        dimension (int): Dimension of the space. Default to 0.
    """

    def __init__(self, dimension=0):
        assert isinstance(dimension, int)
        assert dimension % 2 == 0
        self.d = dimension
        self.r_max = None

    def transform(self, f, k, method, **params):
        r"""Evaluate the Fourier transform of the radially symmetric function :math:`f` at :math:`k` using the correspondence with the Hankel transform.

        Args:
            f (callable): Function to transform.

            k (scalar or numpy.ndarray): Point(s) where the Fourier transform is to be evaluated.

            method (str): Name of the method used to compute the underlying Hankel transform: ``"Ogata"`` (see :cite:`Oga05`) or ``"BaddourChouinard"`` (see :cite:`BaCh15`).

        Keyword Args:
            params (dict):

                - If ``method="BaddourChouinard"`` (see :cite:`BaCh15`):

                    - r_max (float): Threshold radius. Considering that the input function :math:`f` to be Hankel transformed is space-limited, then ``r_max`` is such that :math:`f(r)=0` for r > r_max.
                    - nb_points (int, optional): Number of quadrature nodes. Defaults to 300.
                - If ``method="Ogata"`` (see :cite:`Oga05`):

                    - r_max (float, optional): Maximum radius on which the input function :math:`f` to be Hankel transformed was evaluated before the interpolation. Parameter used to conclude a lower bound on :math:`k` on which :math:`f` to be Hankel transformed. Defaults to None.

                    - step_size (float, optional): Step size of the discretization scheme. Defaults to 0.01.

                    - nb_points (int, optional): Number of quadrature nodes. Defaults to 300.


        Returns:
            tuple (np.array, np.array):
                - k: Point(s) where the Fourier transform is to be evaluated.
                - F_k: Fourier transform of ``f`` at ``k``.

        .. proof:definition::

            The Hankel transform :math:`\mathcal{H}_{\nu}` of order :math:`\nu` of :math:`f` is defined by

            .. math::

                \mathcal{H}_{\nu -1}(f)(k) = \int_0^\infty f(r) J_{\nu}(kr)r \mathrm{d}k,

            where :math:`J_{\nu}` is the Bessel function of first kind.

            The d dimensional Fourier transform :math:`\mathcal{F}` of the radially symmetric function :math:`f` at :math:`k` could be defined using the Hankel transform of :math:`x \rightarrow x^{d/2 -1}f(x)` of order :math:`d/2 -1` as follows,

            .. math::

                k^{d/2-1} \mathcal{F}[f](k)
                = (2 \pi)^{d/2}
                \int_{0}^{+\infty}
                    r^{d/2-1}
                    f(r)
                    J_{d/2-1}(kr)
                    r
                    \mathrm{d}r
                = (2 \pi)^{d/2}
                \mathcal{H}_{d/2-1}[\cdot^{d/2-1} f(\cdot)](k).
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
        hankel_transform = hankel_transformer[method]
        return hankel_transform(order)


class HankelTransform:
    r"""Compute the `Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform>`_ of order :math:`\nu`.

    Args:
        order (int, optional): Order of the Hankel transfrom. Defaults to 0.

    .. seealso::

        - :py:class:`~structure_factor.transforms.RadiallySymmetricFourierTransform`,
        - :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`,
        - :py:class:`~structure_factor.transforms.HankelTransformOgata`.
    """

    def __init__(self, order):
        assert order == np.floor(order)
        self.order = order


class HankelTransformBaddourChouinard(HankelTransform):
    r"""Compute the Hankel transform, using the method of :cite:`BaCh15` considering that the input function is space-limited, i.e., :math:`f(r)=0` for :math:`r>r_{max}`.

    Args:
        order (int, optional): Order of the Hankel transfrom. Defaults to 0.

    .. seealso::

        - `MatLab code of Baddour Chouinard <https://openresearchsoftware.metajnl.com/articles/10.5334/jors.82/>`_,
        - `Pyhank Python package <https://pypi.org/project/pyhank/>`_.
    """

    def __init__(self, order=0):

        super().__init__(order=order)
        self.bessel_zeros = None
        self.r_max = None  # R in :cite:`BaCh15` Section 4.B
        self.transformation_matrix = None  # Y in :cite:`BaCh15` Section 6.A

    def compute_transformation_parameters(self, r_max, nb_points):
        r"""Compute parameters involved in the computation of the corresponding Hankel-type transform using the discretization scheme of :cite:`BaCh15`.

        The following object's attributes are defined

        - :py:attr:`~structure_factor.transforms.HankelTransformBaddourChouinard.bessel_zeros`
        - :py:attr:`~structure_factor.transforms.HankelTransformBaddourChouinard.r_max`
        - :py:attr:`~structure_factor.transforms.HankelTransformBaddourChouinard.transformation_matrix`

        Args:
            r_max (float): Threshold radius. Considering that the input function :math:`f` to be Hankel transformed is space-limited, then ``r_max`` satisfies :math:`f(r)=0` for r > r_max.

            nb_points (int): Number of quadrature nodes.
        """
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
        r"""Compute the Hankel transform of ``f`` at ``k``.

        Args:
            f (callable): Function to be Hankel transformed.

            k (np.ndarray, optional): Points of evaluation of the Hankel transform. Defaults to None.

                - If ``k`` is None (default), then ``k = self.bessel_zeros[:-1] / self.r_max`` derived from :py:meth:`~structure_factor.transforms.HankelTransformBaddourChouinard.compute_transformation_parameters`.
                - If ``k`` is provided, the Hankel transform is first computed at the above k values (case k is None), then interpolated using :py:func:`scipy.interpolate.interp1d` with ``interpolation_params`` and finally evaluated at the provided ``k`` values.

        Keyword Args:
            interpolation_params (dict): Keyword arguments of :py:func:`scipy.interpolate.interp1d`.

        Returns:
            tuple (scalar or np.array, scalar or np.array): ``k`` and the evaluations of the Hankel transform of ``f`` at ``k``.
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
            interpolation_params.setdefault("fill_value", "extrapolate")
            interpolation_params.setdefault("kind", "cubic")
            ht = interpolate.interp1d(_k, ht_k, **interpolation_params)
            return k, ht(k)
        return _k, ht_k


class HankelTransformOgata(HankelTransform):
    r"""Compute the Hankel transform using Ogata quadrature :cite:`Oga05`, (Section 5).

    Args:
        order (int, optional): Order of the Hankel transfrom. Defaults to 0.

    .. seealso::

        - `Hankel Python package <https://joss.theoj.org/papers/10.21105/joss.01397>`_
    """

    def __init__(self, order=0):
        super().__init__(order=order)
        self.nodes, self.weights = None, None
        self.k_min = None

    def compute_transformation_parameters(
        self, r_max=None, nb_points=300, step_size=0.01
    ):
        """Compute the quadrature nodes and weights used by :cite:`Oga05` (Equation (5.2)), to evaluate the corresponding Hankel-type transform.

        Args:
            r_max (float, optional): Maximum radius on which the input function :math:`f`, to be Hankel transformed, was evaluated before the interpolation. Parameter used to conclude a lower bound on :math:`k` on which :math:`f` to be Hankel transformed. Defaults to None.

            step_size (float, optional): Step size of the discretization scheme. Defaults to 0.01.

            nb_points (int, optional): Number of quadrature nodes. Defaults to 300.

        Returns:
            tuple (np.ndarray, np.ndarray): Quadrature nodes and weights.
        """
        n = self.order
        h = step_size
        N = nb_points
        self.r_max = r_max
        t = bessel1_zeros(n, N)
        weights = bessel2(n, t) / bessel1(n + 1, t)  # Equation (1.2)
        t *= h / np.pi  # Equivalent of xi variable
        weights *= self._d_psi(t)
        nodes = (np.pi / h) * self._psi(t)  # Change of variable Equation (5.1)
        self.nodes, self.weights = nodes, weights
        if r_max is not None:
            self.k_min = utils._compute_k_min(r_max=r_max, step_size=step_size)
        return nodes, weights

    def transform(self, f, k):
        r"""Compute the Hankel transform of ``f`` evaluated at ``k``, following the work of :cite:`Oga05` (Section 5).

        Args:
            f (callable): Function to be Hankel transformed.

            k (np.array, optional): Points of evaluation of the Hankel transform. Defaults to None.

        Returns:
            tuple(np.ndarray, np.ndarray):
                - k: Points of evaluation of the Hankel transform.
                - H_k: Evaluations of the Hankel transform of ``f`` on ``k``.

        .. important::

            Please call :py:meth:`HankelTransformOgata.compute_transformation_parameters` to define quadrature attributes :py:attr:`~structure_factor.transforms.Ogata.nodes` and :py:attr:`~structure_factor.transforms.Ogata.weights`, before applying :py:meth:`HankelTransformOgata.compute_transform`.
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
        """Change of variable used by :cite:`Oga05` Equation (5.1)."""
        return t * np.tanh((0.5 * np.pi) * np.sinh(t))

    @staticmethod
    def _d_psi(t):
        """Change of variable used by :cite:`Oga05` Equation (5.1)."""
        threshold = 3.5  # threshold outside of which psi' plateaus to -1, 1
        out = np.sign(t)
        mask = np.abs(t) < threshold
        x = t[mask]
        out[mask] = np.pi * x * np.cosh(x) + np.sinh(np.pi * np.sinh(x))
        out[mask] /= 1.0 + np.cosh(np.pi * np.sinh(x))
        return out
