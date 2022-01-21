from structure_factor.structure_factor import StructureFactor
import numpy as np
from structure_factor.point_pattern import PointPattern

#! Work for DSE actually


class SummeryStatistics:
    def __init__(self, list_point_pattern):
        self.list_point_pattern = list_point_pattern  # list of PointPattern
        self.s = len(list_point_pattern)  # number of sample

    def sample_mean(self, estimator, **params):
        list_point_pattern = self.list_point_pattern
        s = self.s
        isinstance(list_point_pattern[0], PointPattern)
        estimator_i = structure_factor_estimator(list_point_pattern[0], estimator)
        k, sf_estimator_i = estimator_i(**params)
        m = sf_estimator_i
        for i in range(1, s):
            estimator_i = structure_factor_estimator(list_point_pattern[i], estimator)
            _, sf_estimator_i = estimator_i(**params)
            m += sf_estimator_i
        m /= s
        return k, m

    def plot_sample_mean(self, k, m, **params):
        pp = self.list_point_pattern[0]
        sf = StructureFactor(pp)
        return sf.plot_tapered_periodogram(k, m, **params)

    #! non optimized implementation
    def sample_mean_variance(self, estimator, **params):
        s = self.s  # number of sample
        list_point_pattern = self.list_point_pattern  # list of point pattern
        k, m = self.sample_mean(estimator, **params)
        n = k.shape[0]  # number of wavevectors
        var = np.zeros_like(m)
        for i in range(0, s):
            estimator_i = structure_factor_estimator(list_point_pattern[i], estimator)
            _, x_i = estimator_i(**params)
            var += np.square(x_i - m)
        var /= s - 1
        ivar = sum(var) / n
        return k, m, var, ivar

    def sample_bias(self, exact_sf, estimator, **params):
        s = self.s  # number of sample
        k, m = self.sample_mean(estimator, **params)
        n = k.shape[0]  # number of wavevectors
        bias = m - exact_sf(k)
        ibias = sum(bias) / n
        return k, bias, ibias

    def sample_statistics(self, exact_sf, estimator, **params):
        k, m, var, ivar = self.sample_mean_variance(estimator, **params)
        n = k.shape[0]  # number of wavevectors
        bias = m - exact_sf(k)
        ibias = sum(bias) / n
        mse = var + bias ** 2
        imse = ivar + ibias ** 2
        return k, m, var, ivar, bias, ibias, mse, imse


def structure_factor_estimator(point_pattern, estimator):
    assert isinstance(point_pattern, PointPattern)
    sf_pointpattern = StructureFactor(point_pattern=point_pattern)
    if estimator == "scattering_intensity":
        return sf_pointpattern.scattering_intensity
    elif estimator == "tapered_periodogram":
        return sf_pointpattern.tapered_periodogram
    elif estimator == "multitapered_periodogram":
        return sf_pointpattern.multitapered_periodogram

    else:
        raise ValueError(
            "Available estimators are:'scattering_intensity', 'tapered_periodogram','multitapered_periodogram' "
        )
