import numpy as np

from hypton.utils import (
    _binning_function,
)


class EffectiveHyperuniform:
    # todo docstring
    # ! What is the purpose of this class ?
    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf):
        self.norm_k = norm_k
        self.sf = sf
        self.fitted_line = None

    def bin_data(self, **params):
        # ! what is the purpose of this method ?
        return _binning_function(self.norm_k.ravel(), self.sf.ravel(), **params)

    def index_H(self, norm_k, sf, i_max=None):
        # todo docstring
        i = len(norm_k) if i_max is None else i_max
        slope, S_0 = np.polyfit(norm_k[:i], sf[:i], deg=1)
        thresh = 1
        S_first_peak = max(thresh, sf[np.argmax(sf > thresh)])
        self.fitted_line = lambda x: slope * x + S_0
        return S_0 / S_first_peak
