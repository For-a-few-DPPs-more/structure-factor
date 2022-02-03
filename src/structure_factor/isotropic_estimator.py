import numba_scipy
import numba_scipy.special
import numpy as np
import scipy.special as sc
from numba import njit
from scipy.spatial.distance import pdist
from structure_factor.spatial_windows import BallWindow

from structure_factor.spatial_windows import UnitBallWindow


def allowed_k_norm(d, r, n):
    """Allowed wavenumbers of Bartlett isotropic estimator.

    Args:
        d ([type]): [description]
        r ([type]): [description]
        n ([type]): [description]

    Returns:
        [type]: [description]
    """
    return sc.jn_zeros(d / 2, n) / r


#! care about case k=0


def bartlett_estimator(point_pattern, k_norm=None, n_allowed_k_norm=60):
    r"""Compute Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` of the point process (isotropic) encapsulated in the ``PointPattern``.

    Args:
        k_norm (np.ndarray, optional): n rows of wavenumbers where the estimator is evaluated. If ``k_norm=None`` (recommended), the estimator will be evaluated on the corresponding set of allowed wavenumbers; In this case, the parameters ``n_allowed_k_norm`` allows to specify the number of allowed wavenumbers. See :py:func:`~structure_factor.isotropic_estimator.allowed_k_norm`. Defaults to None.
        n_allowed_k_norm(int, optional): Specifies the number of allowed wavenumbers to be used. Used only when ``k_norm=None``. Default to 60.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray):
            - k: Wavenumber(s) on which Bartlett's isotropic estimator has been evaluated.
            - estimation: Evaluation(s) of Bartlett's isotropic estimator at ``k``.
    """
    window = point_pattern.window
    d = window.dimension
    unit_ball = UnitBallWindow(np.zeros(d))

    X = np.atleast_2d(point_pattern.points)
    norm_xi_xj = pdist(X, metric="euclidean")

    # allowed wavenumbers
    if k_norm is None:
        if not isinstance(window, BallWindow):
            raise TypeError(
                "Window must be an instance of BallWindow. Hint: use PointPattern.restrict_to_window."
            )
        k_norm = allowed_k_norm(d=d, r=window.radius, n=n_allowed_k_norm)

    estimator = np.zeros_like(k_norm)
    order = float(d / 2 - 1)
    isotropic_estimator_njit(estimator, k_norm, norm_xi_xj, order)

    surface, volume = unit_ball.surface, window.volume
    rho = point_pattern.intensity
    estimator *= (2 * np.pi) ** (d / 2) / (surface * volume * rho)
    estimator += 1

    return k_norm, estimator


@njit("double(double, double)")
def bessel1_njit(order, x):
    if order == 0.0:
        return sc.j0(x)
    if order == 1.0:
        return sc.j1(x)
    return sc.jv(order, x)


@njit("void(double[:], double[:], double[:], double)")
def isotropic_estimator_njit(result, k_norm, norm_xi_xj, order):
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
