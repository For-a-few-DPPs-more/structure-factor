import numpy as np

from hypton.utils import (
    _binning_function,
)


class EffectiveHyperuniform:
    # ! What is the purpose of this class ?
    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf):
        self.norm_k = norm_k
        self.sf = sf
        self.fitted_line = None

    def bin_data(self, **params):
        # ! what is the purpose of this method ?
        return _binning_function(self.norm_k.ravel(), self.sf.ravel(), **params)

    def index_H(self, norm_k, sf, norm_k_stop=None):
        if norm_k_stop is not None:
            norm_k_list = list(norm_k.ravel())
            index = min(
                range(len(norm_k_list)), key=lambda i: abs(norm_k_list[i] - norm_k_stop)
            )  # index of the closest value to k_stop in norm_k
        i = len(norm_k) if norm_k_stop is None else index
        fitting_param = np.polyfit(norm_k[:i], sf[:i], deg=1)
        S_0 = fitting_param[-1]
        thresh = 1
        S_first_peak = max(thresh, sf[np.argmax(sf > thresh)])
        print(S_first_peak)
        self.fitted_line = lambda x: fitting_param[0] * x + S_0
        return S_0 / S_first_peak
