import numba_scipy
import numba_scipy.special
import numpy as np
import scipy.special as sc
from numba import njit
from scipy.spatial.distance import pdist

from structure_factor.spatial_windows import UnitBallWindow


def isotropic_estimator(point_pattern, norm_k=None):
    window = point_pattern.window
    d = window.dimension
    unit_ball = UnitBallWindow(np.zeros(d))

    X = np.atleast_2d(point_pattern.points)
    norm_xi_xj = pdist(X, metric="euclidean")
    # K = np.atleast_2d(k)
    # norm_k = np.linalg.norm(K, axis=1)
    if norm_k == None:
        norm_k = sc.jn_zeros(d / 2, 200) / window.radius

    estimator = np.zeros_like(norm_k)
    order = float(d / 2 - 1)
    isotropic_estimator_njit(estimator, norm_k, norm_xi_xj, order)

    surface, volume = unit_ball.surface, window.volume
    rho = point_pattern.intensity
    estimator *= (2 * np.pi) ** (d / 2) / (surface * volume * rho)
    estimator += 1

    return norm_k, estimator


@njit("double(double, double)")
def bessel1_njit(order, x):
    if order == 0.0:
        return sc.j0(x)
    if order == 1.0:
        return sc.j1(x)
    return sc.jv(order, x)


@njit("void(double[:], double[:], double[:], double)")
def isotropic_estimator_njit(result, norm_k, norm_xi_xj, order):
    for k in range(len(norm_k)):
        sum_k = 0.0
        for ij in range(len(norm_xi_xj)):
            k_xi_xj = norm_k[k] * norm_xi_xj[ij]
            tmp = bessel1_njit(order, k_xi_xj)
            if order > 0:
                k_xi_xj = k_xi_xj ** order
                tmp /= k_xi_xj
            sum_k += tmp
        result[k] = 2 * sum_k


# old version without numba
def select_periodogram_isotropic(debiased, direct):
    return isotropic_bartlett_estimator


def isotropic_bartlett_estimator(k_norm, point_pattern, taper=None):
    # ! the current implem may take some time when there are a lot of wave vectors and points
    window = point_pattern.window
    #! fi window doenst exists ?
    d = window.dimension
    unit_ball = UnitBallWindow(np.zeros(d))

    X = np.atleast_2d(point_pattern.points)

    norm_x_y = pdist(X, metric="euclidean")  # ||x_i - x_j|| for i < j

    k_xy = np.multiply.outer(k_norm, norm_x_y)
    order = d / 2 - 1
    J_k_xy = bessel1(order, k_xy)

    estimated_sf_k = np.zeros_like(k_norm)
    # for i, k_ in enumerate(k_norm):
    #     estimated_sf_k[i] = bessel1(order, k_ * norm_x_y).sum()
    if order > 0:
        np.power(k_xy, order, out=k_xy)
        J_k_xy /= k_xy
    np.sum(J_k_xy, axis=1, out=estimated_sf_k)
    estimated_sf_k *= 2  # sum_{i neq j} = 2 sum_{i<j}

    surface, volume = unit_ball.surface, window.volume
    rho = point_pattern.intensity
    estimated_sf_k *= (2 * np.pi) ** (d / 2) / (surface * volume * rho)
    estimated_sf_k += 1

    return k_norm, estimated_sf_k
