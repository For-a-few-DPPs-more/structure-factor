import warnings
import numpy as np
from structure_factor.structure_factor import StructureFactor
from structure_factor.spatial_windows import BoxWindow, BallWindow
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)
from scipy.stats import poisson


def subwindows_list(
    window, type="BoxWindow", param_0=None, param_max=None, params=None
):
    d = window.dimension
    window_param_max = _subwindow_param_max(window, type)
    if type not in ["BoxWindow", "BallWindow"]:
        raise ValueError(
            "The available subwindow types are BallWindow or BoxWindow. Hint: the parameter corresponding to the type must be 'BallWindow' or 'BoxWindow'. "
        )
    # subwindows list of parameters
    if params is None:
        if param_0 is None:
            raise ValueError(
                "The minimum window parameter is mandatory. Hint: specify the minimum window parameter."
            )
        # from params0 till param_max with space 1
        if param_max is None:
            params = np.arange(param_0, window_param_max)
    else:
        if max(params) > window_param_max:
            raise ValueError(
                f"The maximum sub-window (parameter={max(params)}) is bigger than the initial window (parameter={param_max}). Hint: Reduce the maximum subwindow parameter. "
            )
    # subwindows list
    if type == "BallWindow":
        subwindows = [BallWindow(center=[0] * d, radius=r) for r in params]
        if d > 1:
            k_list = [
                allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=1)
                for r in params
            ]
        else:
            k_list = None

    else:
        subwindows = [BoxWindow(bounds=[[-l / 2, l / 2]] * d) for l in params]
        k_list = [np.full((1, d), fill_value=2 * np.pi / l) for l in params]
    return subwindows, k_list


def k_list(d, subwindows_params, estimator_type):
    # non-isotropic case
    if estimator_type in ["scattering_intensity", "tapered_estimator"]:
        k_list = [np.full((1, d), fill_value=2 * np.pi / l) for l in subwindows_params]
    # isotropic case
    else:
        k_list = [
            allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=1)
            for r in subwindows_params
        ]
    return k_list


# todo add test
def multiscale_estimator(
    point_pattern,
    estimator,
    k_list,
    subwindows_list,
    mean_poisson,
    m=None,
    proba_list=None,
    verbose=True,
    **kwargs,
):

    # r.v. threshold
    m_thresh = m_threshold(
        window_min=subwindows_list[0],
        window_max=subwindows_list[-1],
    )
    # r.v. M
    if m is None:
        m = _poisson_rv(mean_poisson, m_thresh, verbose=verbose)
    else:
        if m > m_thresh:
            warnings.warn(
                message=f"The random variable M exceed the allowed threshold {m_thresh}."
            )
    m = int(m)
    # proba list
    if proba_list is None:
        proba_list = 1 - (
            poisson.cdf(k=range(m), mu=mean_poisson)
            - poisson.pmf(k=range(m), mu=mean_poisson)
        )
    else:
        if len(proba_list) < m:
            raise ValueError(f"The proba list should contains {max(m)} elements.")
        proba_list = proba_list[:m]

    # k and subwindows list
    if len(subwindows_list) != len(k_list):
        raise ValueError(
            "The number of wavevectors (or wavenumber) k should be equal to the number of subwindows, since each k is associated to a subwindow."
        )
    if len(subwindows_list) < m:
        raise ValueError(
            f"The number of subwindows {len(subwindows_list)} should be at least equal to the random variable M= {m}."
        )
    subwindows_list = subwindows_list[:m]
    k_list = k_list[:m]

    # approximated s_k_min list
    s_k_min_list = multiscale_estimator_core(
        point_pattern=point_pattern,
        subwindows_list=subwindows_list,
        k_list=k_list,
        estimator=estimator,
        **kwargs,
    )

    y_list = [min(np.array([1]), s) for s in s_k_min_list]
    z = coupled_sum_estimator(y_list, proba_list)

    return z


def m_threshold(window_min, window_max):
    if isinstance(window_min, BoxWindow):
        subwindow_type = "BoxWindow"
    else:
        subwindow_type = "BallWindow"
    param_max = _subwindow_param_max(window_max, type=subwindow_type)
    param_min = _subwindow_param_max(window_min, type=subwindow_type)
    if param_max < param_min:
        raise ValueError("window_min should be bigger than window_max.")
    m_threshold = int(param_max - param_min)
    return m_threshold


def multiscale_estimator_core(
    point_pattern, subwindows_list, k_list, estimator, **kwargs
):
    point_pattern_list = [point_pattern.restrict_to_window(w) for w in subwindows_list]
    estimated_sf_k_list = [
        _select_structure_factor_estimator(
            point_pattern=p, estimator=estimator, k=k, **kwargs
        )
        for p, k in zip(point_pattern_list, k_list)
    ]
    return estimated_sf_k_list


def coupled_sum_estimator(y_list, proba_list):
    y_list_with_0 = np.append(0, y_list)  # 0 first element of the list
    y_pairwise_diff = np.array(
        [t - s for s, t in zip(y_list_with_0[:-1], y_list_with_0[1:])]
    )
    y_pairwise_diff = y_pairwise_diff / np.array(proba_list)
    z = np.sum(y_pairwise_diff)
    return z


def _select_structure_factor_estimator(point_pattern, estimator, k, **kwargs):
    sf = StructureFactor(point_pattern)
    if estimator == "scattering_intensity":
        _, estimated_sf_k = sf.scattering_intensity(k=k, **kwargs)
    elif estimator == "tapered_estimator":
        _, estimated_sf_k = sf.tapered_estimator(k=k, **kwargs)
    elif estimator == "bartlett_isotropic_estimator":
        _, estimated_sf_k = sf.bartlett_isotropic_estimator(k_norm=k, **kwargs)
    elif estimator == "quadrature_estimator_isotropic":
        _, estimated_sf_k = sf.quadrature_estimator_isotropic(k_norm=k, **kwargs)
    else:
        raise ValueError(
            "Available estimators: 'scattering_intensity', 'tapered_estimator', 'bartlett_isotropic_estimator', 'quadrature_estimator_isotropic'. "
        )
    return estimated_sf_k


def _subwindow_param_max(window, type="BoxWindow"):
    # window parameter
    if isinstance(window, BallWindow):
        if type == "BallWindow":
            param_max = window.radius
        else:
            param_max = window.radius * 2 / np.sqrt(2)
            # length side of the BoxWindow
    elif isinstance(window, BoxWindow):
        if type == "BallWindow":
            param_max = np.min(np.diff(window.bounds)) / 2
        else:
            param_max = np.min(np.diff(window.bounds))
    return param_max


def _poisson_rv(mean_poisson, threshold, verbose=True):

    m = poisson.rvs(mu=mean_poisson, size=1)
    while m > threshold:
        if verbose:
            print("Re-sample M; current M= ", m, ", threshold=", threshold)
        m = int(poisson.rvs(mu=mean_poisson, size=1))
    return m
