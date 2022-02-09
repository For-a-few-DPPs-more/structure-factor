from structure_factor.structure_factor import StructureFactor
import numpy as np
from structure_factor.point_pattern import PointPattern
from multiprocessing import Pool, freeze_support
from functools import partial
import structure_factor.utils as utils
from structure_factor.isotropic_estimator import allowed_k_norm
from structure_factor.pair_correlation_function import PairCorrelationFunction as pcf
from structure_factor.hyperuniformity import Hyperuniformity


#! Work for DSE actually


class SummaryStatistics:
    def __init__(self, list_point_pattern):
        self.list_point_pattern = list_point_pattern  # list of PointPattern
        self.s = len(list_point_pattern)  # number of sample

    def sample_approximation(self, estimator, core_num=7, **params):
        """approximate the structure factor of list of point process using a DSI or Bartlett isotropic estimator"""
        list_point_pattern = self.list_point_pattern
        isinstance(list_point_pattern[0], PointPattern)
        freeze_support()
        with Pool(core_num) as pool:
            approximations = pool.map(
                partial(apply_estimator, estimator=estimator, **params),
                list_point_pattern,
            )
        return approximations

    def sample_integral_approximation(self, pcf_interpolate_list, **params):
        """approximate the structure factor of list of point process using a hankel transform"""
        s = len(pcf_interpolate_list)
        estimator_list = [
            apply_estimator(
                self.list_point_pattern[0],
                estimator="hankel_quadrature",
                pcf=pcf_interpolate_list[i],
                **params
            )
            for i in range(s)
        ]
        return estimator_list

    def sample_pcf_approximation(self, method, core_num=7, **params):
        """approximate the pair correlation function of a list of sample from a point process."""
        list_point_pattern = self.list_point_pattern
        isinstance(list_point_pattern[0], PointPattern)
        freeze_support()
        with Pool(core_num) as pool:
            pcf_list = pool.map(
                partial(pcf.estimate, method=method, **params),
                list_point_pattern,
            )
        return pcf_list

    def sample_statistics(self, k=None, k_norm=None, approximation=None, exact=None):
        """mean, variance, bias"""
        s = len(approximation)  # number of sample

        if k is not None:
            k_norm = utils.norm_k(k)
        # n = k_norm.shape[0]  # number of k
        m = sum(approximation) / s
        # print(m.shape)
        var = np.square(approximation - m)
        # print(var.shape)
        var = np.sum(var, axis=0)
        # print(var.shape)
        var /= s
        # m_var = sum(var) / n # mean of var
        bias = m - exact(k_norm)
        # m_bias = sum(bias ** 2) / n  # mean of bias square
        mse = var + bias ** 2
        # mse_1 = mse[:-1]
        return m, var, bias, mse

    def plot_sample_mean(self, k, m, **params):
        pp = self.list_point_pattern[0]
        sf = StructureFactor(pp)
        if k.ndim == 2:
            return sf.plot_tapered_periodogram(k, m, **params)
        if k.ndim == 1:
            return sf.plot_isotropic_estimator(k, m, **params)


def structure_factor_estimator(point_pattern, estimator):
    assert isinstance(point_pattern, PointPattern)
    sf_pointpattern = StructureFactor(point_pattern=point_pattern)
    if estimator == "scattering_intensity":
        return sf_pointpattern.scattering_intensity
    elif estimator == "tapered_periodogram":
        return sf_pointpattern.tapered_periodogram
    elif estimator == "multitapered_periodogram":
        return sf_pointpattern.multitapered_periodogram
    elif estimator == "bartlett_isotropic_estimator":
        return sf_pointpattern.bartlett_isotropic_estimator
    elif estimator == "hankel_quadrature":
        return sf_pointpattern.hankel_quadrature
    else:
        raise ValueError(
            "Available estimators are:'scattering_intensity', 'tapered_periodogram','multitapered_periodogram', 'bartlett_isotropic_estimator', 'hankel_quadrature' "
        )


def apply_estimator(point_pattern, estimator, **params):
    estimator = structure_factor_estimator(point_pattern, estimator)
    _, approximation = estimator(**params)
    return approximation


def get_k(point_pattern, **params):
    k = params.get("k")
    if k is None:
        d = point_pattern.dimension
        window = point_pattern.window
        L = np.diff(window.bounds)
        k = utils.allowed_wave_vectors(d, L, **params)
    return k


def get_k_norm(point_pattern, **params):
    k_norm = params.get("k_norm")
    if k_norm is None:
        n = params.get("n_allowed_k_norm")
        d = point_pattern.dimension
        window = point_pattern.window
        r = window.radius
        k_norm = allowed_k_norm(d=d, r=r, n=n)
    return k_norm


# pair correlation function of list


def pcf_interpolate_list(r_list, pcf_list, **params):
    s = len(r_list)
    pcf_interpolate_list = [
        pcf.interpolate(r_list[i], pcf_list[i], params) for i in range(s)
    ]
    return pcf_interpolate_list


# hyperuniformity study
def hyperunifomity_study(k_norm, s_k_norm, estimator):
    """chose test of hyperuniformity"""
    hp = Hyperuniformity(k_norm, s_k_norm)
    if estimator == "effective":
        return hp.effective_hyperuniformity
    elif estimator == "class":
        return hp.hyperuniformity_class
    else:
        raise ValueError("Available estimators are: 'effective' and 'class'. ")


def apply_hyperuniformity_estimate(k_norm, s_k_norm, estimator, **params):
    """Apply hyperuniformity test on a sample from a point process using the approximations of its structure factor"""
    estimator = hyperunifomity_study(k_norm, s_k_norm, estimator)
    results, _ = estimator(**params)
    return results


def sample_hyperuniformity(k_norm, sample_s_k_norm, estimator, **params):
    """Study the hyperuniformity of a list of sample of a point process using the list of approximations of there structure factor ``sample_s_k_norm``.

    Args:
        k_norm (np.ndarray): wavenumbers.
        sample_s_k_norm (list): approximation of the structure factor of s samples of a point process.
        estimator ([type]): "effective", or "class". Studying the index H of the power decay alpha of the estimations.

    Returns:
        [type]: list of H or alpha.
    """
    sample_size = len(sample_s_k_norm)
    results = [
        apply_hyperuniformity_estimate(k_norm, sample_s_k_norm[i], estimator, **params)
        for i in range(sample_size)
    ]
    return results


# sample variance
def sample_variance(x):
    """Compute the sample variance of the list of estimations `x`

    Args:
        x (list): list of estimations
    """
    s = len(x)
    m = sum(x) / s
    # print(m.shape)
    var = np.square(x - m)
    # print(var.shape)
    var = np.sum(var, axis=0)
    # print(var.shape)
    var /= s
    return var, m


def list_vertical_sample_mean(k, list_s_k):
    list_s_k_new = []
    for i in range(len(list_s_k)):
        k_new, s_k_new = vertical_sample_mean(k, list_s_k[i])
        list_s_k_new.append(s_k_new)
    return k_new, list_s_k_new


def vertical_sample_mean(k, s_k):
    "Given repeated elements in x with diffrent values in s_k, this fct find the mean of s_k associated to each unique element of k, and return new k without the repetitions and new s_k with the containing the mean values"
    data = np.column_stack((k, s_k))
    results = {}
    for entry in sorted(data, key=lambda t: t[0]):
        try:
            results[entry[0]] = results[entry[0]] + [entry[1]]
        except KeyError:
            results[entry[0]] = [entry[1]]
    matrix_results = np.array([[key, np.mean(results[key])] for key in results.keys()])
    k_new = matrix_results[:, 0]
    s_k_new = matrix_results[:, 1]
    return k_new, s_k_new
