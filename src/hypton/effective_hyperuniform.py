import numpy as np
from scipy.linalg.misc import norm
from scipy.signal import find_peaks
from hypton.utils import (
    _binning_function,
)
import scipy.optimize as optimization


class EffectiveHyperuniform:
    r"""Test of effective hyperuniformity of a point process :math:`\mathcal{X} \subset \mathbb{R}^2` using the index :math:`H` defined by:

    .. math::
        H = \frac{\hat{S}(\mathbf{0})}{S(\mathbf{k}_{peak})},

    where :math:`S` is the structure factor of :math:`\mathcal{X}`, :math:`\hat{S}(\mathbf{0})` is a linear extrapolation of the structure factor to :math:`\mathbf{k}=\mathbf{0}` and :math:`\mathbf{k}_{peak}` is the location of the first dominant peak value of the structure factor.
    :math:`\mathcal{X}` is sad to be effectively hyperuniform if :math:`H \leq 10^{-3}`.

    .. note::

                Typical usage: the structure factor is estimated using one of the method provided by the class :py:class:`.StructureFactor`, this estimations are average over bins using :py:meth:`bin_data`.

    """

    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf):
        """

        Args:
            norm_k (numpy.ndarray): vector of wave lengths (i.e. norms of the waves) on which the structure factor is provided.

            sf (numpy.ndarray): vector of the scattering intensity associated to ``norm_k``.
        """
        self.norm_k = norm_k
        self.sf = sf
        self.fitted_line = None
        self.i_first_peak = None

    def bin_data(self, **params):
        """Regularization of the estimation of the structure factor, by spliting the vector ``norm_k`` into bins and we average the associated values of the vector ``sf`` and derive the standard deviation over each bins.

        Args:
            params(dict): parameters associated to _binning_function.

        Returns:
            bin_centers: vector of centers of the bins representing the new vector ``norm_k``

            bin_mean: vector of means of the scattering intensity ``sf``over the bins, representing the new vector ``sf``.

            bin_std: vector of standard deviations corresponding to ``bin_mean``.

        .. seealso::
            :py:func:`~utils._binning_function`

        """
        return _binning_function(self.norm_k.ravel(), self.sf.ravel(), **params)

    def index_H(self, norm_k, sf, std=None, norm_k_stop=None):
        """Estimation of the effective hyperuniformity of a point process :math:`\mathcal{X} \subset \mathbb{R}^2` using the index :math:`H`.

        .. important::

             To find the numerator :math:`\hat{S}(\mathbf{0})` of the index math:`H`, we fit a line using a linear regression with least square fit of the approximated structure factor associated to the values in ``norm_k`` less than ``nom_k_{stop}`` (which must be chosen before the stabilization of the structure factor around 1).
             If the standard deviation of the approximated vector of structure factor ``sf`` is provided via the argument ``std`` then they will be considered while fitting the line.


        Args:
            norm_k (numpy.1darray): vector of wave lengths (i.e. norms of the waves) on which the structure factor is provided.

            sf (numpy.1darray): vector of the scattering intensity associated to ``norm_k``.

            std (numpy.1darray, optional): vector of standard deviation associated to ``sf``. Defaults to None.

            norm_k_stop (float, optional): the bound on ``norm_k`` used for the linear regression. Defaults to None.

        Returns:
            the index :math:`H` and the standard deviations of numerator of :math:`H`.
        """

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
        self.fitted_line = lambda x: fitting_params[1] * x + S_0
        thresh = 1
        i_peak, _ = find_peaks(sf, height=thresh)

        if list(i_peak):
            self.i_first_peak = max(i_peak[0], 1)
            S_first_peak = sf[self.i_first_peak]
        else:
            S_first_peak = 1

        return S_0 / S_first_peak, std_intercept
