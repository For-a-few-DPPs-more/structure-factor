import numpy as np

from structure_factor.utils import (
    _binning_function,
)


class EffectiveHyperuniform:
    # ! What is the purpose of this class ?
    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, k_norm, sf):
        self.k_norm = k_norm
        self.sf = sf

    def bin_data(self, **params):
        # ! what is the purpose of this method ?
        return _binning_function(self.k_norm, self.sf, **params)

    def index_H(self, k_norm, sf, i_max=None):
        i = len(k_norm) if i_max is None else i_max
        S_0 = np.polyfit(k_norm[:i], sf[:i], deg=1)[-1]
        thresh = 1
        S_first_peak = max(thresh, sf[np.argmax(sf > thresh)])
        return S_0 / S_first_peak
