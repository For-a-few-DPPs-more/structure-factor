from structure_factor.utils import (
    _binning_function,
    _lsf,
)


class EffectiveHyperuniform:
    def __init__(self, norm_k, sf):
        self.norm_k = norm_k
        self.sf = sf

    def bin_data(self, **binning_params):
        norm_k = self.norm_k
        sf = self.sf
        bin_centers, bin_mean, bin_std = _binning_function(norm_k, sf, **binning_params)
        return bin_centers, bin_mean

    def H(self, norm_k, sf, stop=None):
        fitted_line = _lsf(norm_k, sf, stop)
        self.fitted_line = fitted_line
        sf_0 = fitted_line(0)
        index_k_peak = sf > 1
        if len(sf[index_k_peak]) > 0:
            sf_k_peak = sf[index_k_peak][0]
        else:
            sf_k_peak = 1
        H = sf_0 / sf_k_peak
        return H
