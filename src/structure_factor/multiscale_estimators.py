import warnings
import numpy as np
from structure_factor.structure_factor import StructureFactor
from structure_factor.spatial_windows import BoxWindow, BallWindow
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)
from scipy.stats import poisson


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


def coupled_sum_estimator(y_list, proba_list):
    y_list_with_0 = np.append(0, y_list)  # 0 first element of the list
    y_pairwise_diff = np.array(
        [t - s for s, t in zip(y_list_with_0[:-1], y_list_with_0[1:])]
    )
    y_pairwise_diff = y_pairwise_diff / np.array(proba_list)
    z = np.sum(y_pairwise_diff)
    return z


def _subwindow_param_max(window, type="Box"):
    # window parameter
    if isinstance(window, BallWindow):
        if type == "Ball":
            param_max = window.radius
        else:
            param_max = window.radius * 2 / np.sqrt(2)
            # length side of the BoxWindow
    elif isinstance(window, BoxWindow):
        if type == "Ball":
            param_max = np.diff(window.bounds)[0] / 2
        else:
            param_max = np.diff(window.bounds)[0]
    return param_max


def _subwindows(window, type="Box", param_0=None, param_max=None, params=None):
    d = window.dimension
    window_param_max = _subwindow_param_max(window, type)
    if type not in ["Box", "Ball"]:
        raise ValueError(
            "The available subwindow types are Ball or Box. Hint: the parameter corresponding to the type must be 'Ball' or 'Box'. "
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
    if type == "Ball":
        subwindows = [BallWindow(center=[0] * d, radius=r) for r in params]
    else:
        subwindows = [BoxWindow(bounds=[[-l / 2, l / 2]] * d) for l in params]
    return subwindows, params


def _k_list(estimator, d, subwindows_params):
    # non-isotropic case
    if estimator in ["scattering_intensity", "tapered_estimator"]:
        k_list = [np.full((1, d), fill_value=2 * np.pi / l) for l in subwindows_params]
    # isotropic case
    else:
        k_list = [
            allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=1)
            for r in subwindows_params
        ]
    return k_list


def _poisson_rv(mean_poisson, threshold, verbose=True):

    m = poisson.rvs(mu=mean_poisson, size=1)
    while m > threshold:
        if verbose:
            print("Re-sample M; current M= ", m, ", threshold=", threshold)
        m = int(poisson.rvs(mu=mean_poisson, size=1))
    return m


def _subwindows_type(estimator):
    if estimator in ["scattering_intensity", "tapered_estimator"]:
        subwindows_type = "Box"
    elif estimator in [
        "bartlett_isotropic_estimator",
        "quadrature_estimator_isotropic",
    ]:
        subwindows_type = "Ball"
    else:
        raise ValueError(
            "Available estimators: 'scattering_intensity', 'tapered_estimator', 'bartlett_isotropic_estimator', 'quadrature_estimator_isotropic'. "
        )
    return subwindows_type


# todo add test
def multiscale_estimator(
    point_pattern,
    estimator,
    k_list=None,
    subwindows_params=None,
    subwindows_param_0=None,
    m=None,
    proba_list=None,
    mean_poisson=None,
    m_threshold=None,
    verbose=False,
    **kwargs,
):

    d = point_pattern.dimension
    window = point_pattern.window
    # subwindow param 0
    if subwindows_params is None:
        if subwindows_param_0 is None:
            raise ValueError(
                "The minimum window parameter is mandatory. Hint: specify the minimum subwindow parameter `subwindows_param_0`."
            )
    else:
        subwindows_param_0 = subwindows_params[0]
    # r.v. threshold
    if m_threshold is None:
        # subwindow param max
        subwindows_param_max = _subwindow_param_max(window, type=type)
        m_threshold = subwindows_param_max - subwindows_param_0
    m_threshold = int(m_threshold)
    # r.v. M
    if m is None:
        m = _poisson_rv(mean_poisson, m_threshold, verbose=verbose)
    else:
        if m > m_threshold:
            warnings.warn(
                message=f"The random variable M exceed the allowed threshold {m_threshold}."
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
    # subwindow
    subwindows_type = _subwindows_type(estimator)
    subwindows, subwindows_params = _subwindows(
        window=window,
        type=subwindows_type,
        param_0=subwindows_param_0,
        param_max=subwindows_param_0 + m,
        params=subwindows_params,
    )

    # k_list
    if k_list is None:
        k_list = _k_list(estimator, d, subwindows_params)
    else:
        if len(subwindows_params) != len(k_list):
            raise ValueError(
                "The number of wavevectors (or wavenumber) k should be equal to the number of subwindows, since each k is associated to a subwindow."
            )
    # approximated s_k_min list
    s_k_min_list = multiscale_estimator_core(
        point_pattern=point_pattern,
        subwindows_list=subwindows,
        k_list=k_list,
        estimator=estimator,
        **kwargs,
    )

    y_list = [min(np.array([1]), s) for s in s_k_min_list]
    z = coupled_sum_estimator(y_list, proba_list)

    return z, m


# def _m_list(mean_poisson, nb_m, threshold, verbose=True):
#     m_list = []
#     for _ in range(nb_m):
#         m = poisson.rvs(mu=mean_poisson, size=1)
#         while m > threshold:
#             if verbose:
#                 print("Re-sample M; current M= ", m, ", threshold=", threshold)
#             m = int(poisson.rvs(mu=mean_poisson, size=1))
#         m_list.append(m)
#     return m_list
