import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from hypton.utils import _binning_function


class EffectiveHyperuniform:
    r"""Test of effective hyperuniformity of a point process :math:`\mathcal{X} \subset \mathbb{R}^2`.

    :math:`\mathcal{X}` is said to be effectively hyperuniform if :math:`H \leq 10^{-3}` where

    .. math::
        H = \frac{\hat{S}(\mathbf{0})}{S(\mathbf{k}_{peak})}\cdot

    - :math:`S` is the structure factor of :math:`\mathcal{X}`,
    - :math:`\hat{S}(\mathbf{0})` is a linear extrapolation of the structure factor at :math:`\mathbf{k}=\mathbf{0}`,
    - :math:`\mathbf{k}_{peak}` is the location of the first dominant peak value of :math:`S`.

    .. note::

        **Typical usage**:

        Estimating the structure factor of a point process by one of the method of the class :py:class:`~.hypton.StructureFactor`, then testing the effective hyperuniformity using :py:meth:`~EffectiveHyperuniform.index_H`.

    .. todo::

        Add bibliographic reference.
    """
    # ? How about EffectiveHyperuniformity instead of EffectiveHyperuniform.
    # ? The later qualifies a class of point processes while former characterizes a property of a point process
    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf):
        """
        Args:
            norm_k (numpy.ndarray): vector of wave lengths (i.e. norms of the waves).

            sf (numpy.ndarray): Evalutation of the structure factor at ``norm_k``.
        """
        self.norm_k = norm_k
        self.sf = sf
        self.fitted_line = None
        self.i_first_peak = None

    def bin_data(self, **params):
        """Regularization of the estimation of the structure factor, by spliting the vector ``norm_k`` into bins and we average the associated values of the vector ``sf`` and derive the standard deviation over each bins.

        Args:
            params(dict): parameters associated to :py:func:`~.hypton.utils._binning_function`.

        Returns:
            bin_centers: vector of centers of the bins representing the new vector ``norm_k``.

            bin_mean: vector of means of the scattering intensity ``sf`` over the bins, representing the new vector ``sf``.

            bin_std: vector of standard deviations corresponding to ``bin_mean``.

        .. seealso::
            :py:func:`~utils._binning_function`

        """
        return _binning_function(self.norm_k.ravel(), self.sf.ravel(), **params)

    #! c'est bizare qu'elle prend norm_k et sf qui sont deja des atribus de la class
    def index_H(self, norm_k, sf, std=None, norm_k_stop=None):
        """Estimation of the effective hyperuniformity of a point process :math:`\mathcal{X} \subset \mathbb{R}^2` using the index :math:`H`.

        .. important::

            To compute the numerator :math:`\hat{S}(\mathbf{0})` of the index :math:`H`, we fit a line using a linear regression with least square fit of the approximated structure factor associated to the values in ``norm_k`` less than ``nom_k_stop`` (which must be chosen before the stabilization of the structure factor around 1).
            If the standard deviation of the approximated vector of structure factor ``sf`` is provided via the argument ``std`` then they will be considered while fitted the line.


        Args:
            norm_k (numpy.1darray): vector of wave lengths (i.e. norms of the waves) on which the structure factor is provided.

            sf (numpy.1darray): vector of the scattering intensity associated to ``norm_k``.

            std (numpy.1darray, optional): vector of standard deviation associated to ``sf``. Defaults to None.

            norm_k_stop (float, optional): the bound on ``norm_k`` used for the linear regression. Defaults to None.

        Returns:
            the index :math:`H` and the standard deviations of numerator of :math:`H`.
        """

        i = len(norm_k)
        if norm_k_stop is not None:
            # index of the closest value to k_stop in norm_k
            i = np.argmin(np.abs(norm_k.ravel() - norm_k_stop))

        # Fit line
        line = lambda x, a, b: a + b * x

        xdata = norm_k[:i]
        ydata = sf[:i]
        sigma = std[:i] if std is not None else None

        (intercept, slope), cov = curve_fit(
            f=line, xdata=xdata, ydata=ydata, sigma=sigma
        )
        # self.fitted_line = lambda x: line(x, intercept, slope)
        self.fitted_line = lambda x: intercept + slope * x

        # Find first peak in structure factor (sf)
        S0 = intercept
        S0_std = np.sqrt(cov[0, 0])

        S_first_peak = 1
        idx_peaks, _ = find_peaks(sf, height=S_first_peak)
        if idx_peaks.size:
            self.i_first_peak = max(idx_peaks[0], 1)
            S_first_peak = sf[self.i_first_peak]

        return S0 / S_first_peak, S0_std
