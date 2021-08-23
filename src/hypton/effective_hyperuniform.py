import numpy as np
from scipy.linalg.misc import norm
from scipy.signal import find_peaks
from hypton.utils import (
    _binning_function,
)
import scipy.optimize as optimization


class EffectiveHyperuniform:
    # todo docstring
    # ! What is the purpose of this class ?
    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf):
        self.norm_k = norm_k
        self.sf = sf
        self.fitted_line = None
        self.i_first_peak = None

    def bin_data(self, **params):
        # ! what is the purpose of this method ?
        return _binning_function(self.norm_k.ravel(), self.sf.ravel(), **params)

    def index_H(self, norm_k, sf, std=None, norm_k_stop=None):

        if norm_k_stop is not None:
            norm_k_list = list(norm_k.ravel())
            index = min(
                range(len(norm_k_list)), key=lambda i: abs(norm_k_list[i] - norm_k_stop)
            )  # index of the closest value to k_stop in norm_k
        i = len(norm_k) if norm_k_stop is None else index
        poly = lambda x, a, b: a + b * x
        if std is not None:
            fitting_params, fitting_cov = optimization.curve_fit(
                f=poly, xdata=norm_k[:i], ydata=sf[:i], sigma=std[:i]
            )
        else:
            fitting_params, fitting_cov = optimization.curve_fit(
                f=poly, xdata=norm_k[:i], ydata=sf[:i]
            )
        std_intercept = np.sqrt(np.diag(fitting_cov))[0]
        S_0 = fitting_params[0]
        print(fitting_params[1])
        self.fitted_line = lambda x: fitting_params[1] * x + S_0
        thresh = 1
        i_peak, _ = find_peaks(sf, height=thresh)

        if list(i_peak):
            self.i_first_peak = max(i_peak[0], 1)
            S_first_peak = sf[self.i_first_peak]
        else:
            S_first_peak = 1

        return S_0 / S_first_peak, std_intercept
