import warnings
import numpy as np
from structure_factor.structure_factor import StructureFactor
from structure_factor.spatial_windows import (
    BoxWindow,
    BallWindow,
    subwindow_parameter_max,
)
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)
from scipy.stats import poisson


def subwindows_list(
    window, subwindows_type="BoxWindow", param_0=None, param_max=None, params=None
):
    d = window.dimension
    # parameter of the biggest possible subwindow contained in `window``
    window_param_max = subwindow_parameter_max(window, subwindows_type)
    # subwindows list of parameters
    if params is None:
        if param_0 is None:
            raise ValueError(
                "The minimum window parameter is mandatory. Hint: specify the minimum window parameter."
            )
        # from param_0 till param_max with unit space
        if param_max is None:
            params = np.arange(param_0, window_param_max)
    else:
        if max(params) > window_param_max:
            raise ValueError(
                f"The maximum sub-window (parameter={max(params)}) is bigger than the father window (parameter={window_param_max}). Hint: Reduce the maximum subwindow parameter. "
            )
    # subwindows and the associated k
    if subwindows_type == "BallWindow":
        subwindows = [BallWindow(center=[0] * d, radius=r) for r in params]
        # check if d is even
        _, mod_ = divmod(d, 2)
        is_even_dimension = mod_ == 0
        if is_even_dimension:
            k_list = [
                allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=1)
                for r in params
            ]
        else:
            k_list = None
            warnings.warn(
                message=f"Isotropic allowed wavenumber are available only when the dimension of the space is an even number (i.e., d/2 is an integer)."
            )

    else:
        subwindows = [BoxWindow(bounds=[[-l / 2, l / 2]] * d) for l in params]
        k_list = [np.full((1, d), fill_value=2 * np.pi / l) for l in params]
    return subwindows, k_list


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
            raise ValueError(f"The proba list should contains at least{m} elements.")
        proba_list = proba_list[:m]

    # k and subwindows list
    if len(subwindows_list) != len(k_list):
        raise ValueError(
            "The number of wavevectors/wavenumber (k) should be equal to the number of subwindows. Each k is associated to a subwindow."
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
    # the r.v. (y_n)_n
    y_list = [min(np.array([1]), s) for s in s_k_min_list]
    z = coupled_sum_estimator(y_list, proba_list)

    return z


def m_threshold(window_min, window_max):
    if isinstance(window_min, BoxWindow):
        subwindow_type = "BoxWindow"
    else:
        subwindow_type = "BallWindow"
    param_max = subwindow_parameter_max(window_max, subwindow_type=subwindow_type)
    param_min = subwindow_parameter_max(window_min, subwindow_type=subwindow_type)
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
            point_pattern=p, estimator=estimator, k=q, **kwargs
        )
        for p, q in zip(point_pattern_list, k_list)
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


def _poisson_rv(mean_poisson, threshold, verbose=True):

    m = poisson.rvs(mu=mean_poisson, size=1)
    while m > threshold:
        if verbose:
            print("Re-sample M; current M= ", m, ", threshold=", threshold)
        m = int(poisson.rvs(mu=mean_poisson, size=1))
    return m
