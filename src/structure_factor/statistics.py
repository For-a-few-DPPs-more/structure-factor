from structure_factor.structure_factor import StructureFactor
import numpy as np
from structure_factor.point_pattern import PointPattern
from multiprocessing import Pool, freeze_support
from functools import partial
import structure_factor.utils as utils
from structure_factor.isotropic_estimator import allowed_k_norm


#! Work for DSE actually


class SummaryStatistics:
    def __init__(self, list_point_pattern):
        self.list_point_pattern = list_point_pattern  # list of PointPattern
        self.s = len(list_point_pattern)  # number of sample

    def sample_approximation(self, estimator, core_num=7, **params):
        list_point_pattern = self.list_point_pattern
        s = self.s
        isinstance(list_point_pattern[0], PointPattern)
        freeze_support()
        with Pool(core_num) as pool:
            approximations = pool.map(
                partial(apply_estimator, estimator=estimator, **params),
                list_point_pattern,
            )
        return approximations

    def sample_statistics(self, k, approximation, exact):
        s = len(approximation)  # number of sample
        n = k.shape[0]  # number of k
        norm_k = utils.norm_k(k)
        m = sum(approximation) / s
        var = np.square(approximation - m)
        var = np.sum(var, axis=0)
        var /= s - 1
        ivar = sum(var) / n
        bias = m - exact(norm_k)
        ibias = sum(bias) / n
        mse = var + bias ** 2
        imse = sum(mse) / n

        return m, var, ivar, bias, ibias, mse, imse

    def plot_sample_mean(self, k, m, **params):
        pp = self.list_point_pattern[0]
        sf = StructureFactor(pp)
        return sf.plot_tapered_periodogram(k, m, **params)


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
    else:
        raise ValueError(
            "Available estimators are:'scattering_intensity', 'tapered_periodogram','multitapered_periodogram' "
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
        L = np.diff(window.bounds[0])
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
