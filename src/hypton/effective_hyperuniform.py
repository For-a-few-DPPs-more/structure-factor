import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from hypton.utils import _binning_function


class EffectiveHyperuniformity:
    r"""Test of effective hyperuniformity of a stationary isotropic (or effectively isotropic) point process :math:`\mathcal{X} \subset \mathbb{R}^2`.

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

    # ! Can't it be reduced to simple function calls to _binning_function and index_H?

    def __init__(self, norm_k, sf, std_sf=None):
        """
        Args:
            norm_k (numpy.ndarray): vector of wave lengths (i.e. norms of the waves).

            sf (numpy.ndarray): Evalutation of the structure factor at ``norm_k``.

            std (numpy.1darray, optional): vector of standard deviation associated to ``sf``. Defaults to None.

        """
        assert isinstance(norm_k, np.ndarray)
        assert isinstance(sf, np.ndarray)
        assert norm_k.shape == sf.shape
        self.norm_k = norm_k
        self.sf = sf
        self.std_sf = std_sf
        self.fitted_line = None
        self.i_first_peak = None

    def bin_data(self, **params):
        """Regularization of the estimation of the structure factor, by spliting the vector ``norm_k`` into bins and we average the associated values of the vector ``sf`` and derive the standard deviation over each bins.

        Args:
            params(dict): parameters associated to :py:func:`~.hypton.utils._binning_function`.

        Returns:
            self.norm_k(np.1darray): vector of centers of the bins representing the new vector ``norm_k``.

            self.sf(np.1darray): vector of means of the scattering intensity ``sf`` over the bins, representing the new vector ``sf``.

            self.std_sf(np.1darray): vector of standard deviations corresponding to ``bin_mean``.

        .. seealso::
            :py:func:`~utils._binning_function`

        """
        self.norm_k, self.sf, self.std_sf = _binning_function(
            self.norm_k.ravel(), self.sf.ravel(), **params
        )
        return self.norm_k, self.sf, self.std_sf

    def index_H(self, norm_k_stop=None):
        """Estimation of the effective hyperuniformity of a point process :math:`\mathcal{X} \subset \mathbb{R}^2` using the index :math:`H`.

        .. important::

            To compute the numerator :math:`\hat{S}(\mathbf{0})` of the index :math:`H`, we fit a line using a linear regression with least square fit of the approximated structure factor associated to the values in ``norm_k`` less than ``nom_k_stop`` (which must be chosen before the stabilization of the structure factor around 1).
            If the standard deviation of the approximated vector of structure factor ``sf`` is provided via the argument ``std`` then they will be considered while fitted the line.


        Args:

            norm_k_stop (float, optional): the bound on ``norm_k`` used for the linear regression. Defaults to None.

        Returns:
            the index :math:`H` and the standard deviations of numerator of :math:`H`.
        """
        norm_k = self.norm_k
        sf = self.sf
        std = self.std_sf

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
        s0 = intercept
        s0_std = np.sqrt(cov[0, 0])

        s_first_peak = 1
        idx_peaks, _ = find_peaks(sf, height=s_first_peak)
        if idx_peaks.size:
            self.i_first_peak = max(idx_peaks[0], 1)
            s_first_peak = sf[self.i_first_peak]

        return s0 / s_first_peak, s0_std
