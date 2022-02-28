"""Collection of functions used to implement a fast version of Bartlett's isotropic estimator."""


import numpy as np
import scipy.special as sc
from numba import njit
from scipy.spatial.distance import pdist

from structure_factor.spatial_windows import BallWindow, UnitBallWindow


#! care about case k=0
def bartlett_estimator(k_norm, point_pattern):
    r"""Compute an estimation of the structure factor of a stationary isotropic point process from one realization encapsulated in ``point_pattern``, evaluated at ``k_norm``.

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying point process.

        k_norm (numpy.ndarray, optional): Array of size :math:`n` corresponding to the wavenumbers where the estimator is to be evaluated. If None (default) and the observation window ``point_pattern.window`` is a :py:class:`~structure_factor.spatial_windows.BallWindow` and the ambient dimension is even, the estimator will be evaluated on the corresponding set of allowed wavenumbers returned by :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm`. Defaults to None.

        n_allowed_k_norm (int, optional): Number of allowed wavenumbers to be used when ``k_norm`` is None. See :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm`. Defaults to 60.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray):
            - k_norm: Wavenumber(s) at which the estimator has been evaluated.
            - estimation: Evaluation(s) of the estimator at ``k_norm``.

    Example:
        .. literalinclude:: code/isotropic_estimator/bartlett_estimator.py
            :language: python

    .. proof:definition::

        The Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` is constructed from a realization :math:`\{\mathbf{x}_i\}_{i=1}^N` of the point process :math:`\mathcal{X}` observed in a ball window :math:`W=B(\mathbf{0}, R)`.

        .. math::

            \widehat{S}_{\mathrm{BI}}(k)
            = 1
            + \frac{ (2\pi)^{d/2} }{\rho |W| \omega_{d-1}}
            \sum_{ \substack{j, q =1 \\ j\neq q } }^{N}
                \frac{1}{(k \|\mathbf{x}_j - \mathbf{x}_q\|_2)^{d/2 - 1}}
                J_{d/2 - 1}(k \|\mathbf{x}_j - \mathbf{x}_q\|_2).

        For more details, we refer to :cite:`HGBLR:22`, (Section 3.2).

    .. seealso::

        - :py:class:`~structure_factor.spatial_windows.BallWindow`
        - :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`
        - :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator_isotropic`
        - :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm`
    """
    if not isinstance(point_pattern.window, BallWindow):
        raise TypeError(
            "The window must be an instance of BallWindow. Hint: use point_pattern.restrict_to_window."
        )
    window = point_pattern.window
    d = window.dimension

    X = np.atleast_2d(point_pattern.points)
    norm_xi_xj = pdist(X, metric="euclidean")

    k_norm = k_norm.astype(float)
    sf_estimated = np.zeros_like(k_norm)
    order = float(d / 2 - 1)
    bartlett_estimator_njit(sf_estimated, k_norm, norm_xi_xj, order)

    surface = UnitBallWindow(np.zeros(d)).surface
    volume = window.volume
    rho = point_pattern.intensity
    sf_estimated *= (2.0 * np.pi) ** (d / 2) / (surface * volume * rho)
    sf_estimated += 1.0

    return sf_estimated


def allowed_k_norm_bartlett_isotropic(dimension, radius, nb_values=60):
    r"""Allowed wavenumbers of the Bartlett isotropic estimator, for a ``d``-dimensional point process observed in a ball window with radius ``radius``.

    .. warning::

        This method is only available when the ambient dimension is even.

    Args:
        dimension (int): Dimension of the underlying space.
        radius (float): Radius of the observation window.
        nb_values (int): Number of required allowed wavenumbers. Defaults to 60.

    Returns:
        numpy.ndarray: Vector of size ``nb_values`` containing the allowed wavenumbers.

    Example:
        .. literalinclude:: code/isotropic_estimator/allowed_k_norm.py
            :language: python

        .. testoutput::

            [0.1915853  0.35077933 0.50867341 0.6661846  0.8235315  0.98079293]

    .. proof:definition::

        The allowed wavenumbers of a realization from a point process :math:`\mathcal{X}` of :math:`\mathbb{R}^d` observed in a ball window :math:`W=B(\mathbf{0}, R)` correspond to

        .. math::

            \left\{
            \frac{x}{R} \in \mathbb{R} \text{ s.t. }  J_{d/2}(x)=0
            \right\}

    .. seealso::

        - :py:func:`~structure_factor.tapered_estimators_isotropic.bartlett_estimator`
    """
    d_2, mod_ = divmod(dimension, 2)
    is_even_dimension = mod_ == 0
    if not is_even_dimension:  # dimension is not even
        raise ValueError(
            "Allowed wavenumber could be used only when the dimension of the space `d` is an even number (i.e., d/2 is an integer)."
        )
    return sc.jn_zeros(d_2, nb_values) / radius


@njit("double(double, double)")
def bessel1_njit(order, x):
    if order == 0.0:
        return sc.j0(x)
    if order == 1.0:
        return sc.j1(x)
    return sc.jv(order, x)


# todo use cuda.jit to decrease computational time https://towardsdatascience.com/python-performance-and-gpus-1be860ffd58d#:~:text=x%5B%201%2C%201%5D)%20//%209-,GPU%20%E2%80%94%203%20ms,-%40numba.cuda.jit


@njit("void(double[:], double[:], double[:], double)")
def bartlett_estimator_njit(result, k_norm, norm_xi_xj, order):
    for k in range(len(k_norm)):
        sum_k = 0.0
        for ij in range(len(norm_xi_xj)):
            k_xi_xj = k_norm[k] * norm_xi_xj[ij]
            tmp = bessel1_njit(order, k_xi_xj)
            if order > 0:
                k_xi_xj = k_xi_xj ** order
                tmp /= k_xi_xj
            sum_k += tmp
        result[k] = 2 * sum_k
