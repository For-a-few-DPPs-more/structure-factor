import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from structure_factor.utils import _bin_statistics


class Hyperuniformity:
    r"""Compute indicator of hyperuniformity of a stationary isotropic (or effectively isotropic) point process :math:`\mathcal{X} \subset \mathbb{R}^d`, given the evaluation of its structure factor.

    This class contains
        - A method for regularizing the approximated structure factor sample, :meth:`bin_data`, consisting on dividing the vector of wavelength into sub-intervals and taking the mean and strandard deviation over each sub-interval.
        - Test of effective hyperuniformity, :meth:`index_H`, consisting on evaluating the index H of hyperuniformity :cite:`Kla+al19` and used to study if the sample is effectively hyperuniform.
        - Test of power decay of the structure factor near zero, :meth:`power_decay`, used to determine the class of hyperuniformity of the sample :cite:`Cos21`.

    .. note::

        **Typical usage**:

            1- Estimating the structure factor of a point process by one of the methods of :py:class:`~structure_factor.structure_factor.StructureFactor`.

            2- Regularize the results using :meth:`bin_data`.

            3- Testing the effective hyperuniformity using :py:meth:`index_H`.

            4- If the test :meth:`index_H` , approve the effective hyperuniformity hypothesis then use :meth:`power_decay` to study the possible power decay  of the structure factor  to zero which specify the class of hyperuniformity.
    """

    def __init__(self, k_norm, sf, std_sf=None):
        """Initialize the object from the pair ``(k, SF(k))`` which corresponds to the norm of the wave vector and the evaluation of the structure factor.

        Args:
            k_norm (numpy.ndarray): vector of wave lengths (i.e. norms of the waves).

            sf (numpy.ndarray): Evalutation of the structure factor at ``k_norm``.

            std (np.ndarray, optional): vector of standard deviation associated to ``sf``. Defaults to None.

        """
        assert isinstance(k_norm, np.ndarray)
        assert isinstance(sf, np.ndarray)
        assert k_norm.shape == sf.shape
        self.k_norm = k_norm
        self.sf = sf
        self.std_sf = std_sf
        self.fitted_line = None
        self.i_first_peak = None

    def bin_data(self, **params):
        """Regularization of the estimated the structure factor sample.

        This method split the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm` into sub-intervals (or bins)  then evaluate over each sub-interval the mean and the standard deviation of the corresponding associated values of the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`.

        Args:
            params(dict): parameters associated to :py:func:`~structure_factor.utils._bin_statistics`.

        Returns:
            tuple(np.ndarray, np.ndarray, np.ndarray):
                - vector of centers of the bins, representing the new vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`.
                - vector of means of the scattering intensity ``sf`` over the bins, representing the new vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`.
                - vector of standard deviations, representing the new vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.bin_mean`.

        .. seealso::

            :py:func:`~structure_factor.utils._bin_statistics`

        Example:

            .. literalinclude:: code/effective_example.py
                :language: python
                :lines: 22-27
        """
        self.k_norm, self.sf, self.std_sf = _bin_statistics(
            self.k_norm.ravel(), self.sf.ravel(), **params
        )
        return self.k_norm, self.sf, self.std_sf

    # ? how about effective_uniformity
    # todo rename k_norm_stop to k_norm_max
    def index_H(self, k_norm_stop=None, **kwargs):
        r"""Estimate the effective hyperuniformity of a stationary isotropic point process :math:`\mathcal{X} \subset \mathbb{R}^2`, as defined by :cite:`Tor18`. # todo add section

        :math:`\mathcal{X}` is said to be effectively hyperuniform if :math:`H \leq 10^{-3}` where

        .. math::
            H = \frac{\hat{S}(\mathbf{0})}{S(\mathbf{k}_{peak})}\cdot

        - :math:`S` is the structure factor of :math:`\mathcal{X}`,
        - :math:`\hat{S}(\mathbf{0})` is a linear extrapolation of the structure factor at :math:`\mathbf{k}=\mathbf{0}`,
        - :math:`\mathbf{k}_{peak}` is the location of the first dominant peak value of :math:`S`.

        .. important::

            To compute the numerator :math:`\hat{S}(\mathbf{0})` of the index :math:`H`, we fit a line using a linear regression with least square fit of the approximated structure factor associated to the values in ``k_norm`` less than ``k_norm_stop`` (which must be chosen before the stabilization of the structure factor around 1).
            If the standard deviation of the approximated vector of structure factor ``sf`` is provided via the argument ``std`` then they will be considered while fitted the line.

        Args:
            k_norm_stop (float, optional): the bound on ``k_norm`` used for the linear regression. Defaults to None.

        Keyword args:
            kwargs (dict): keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            tuple(float, float): index :math:`H` and the standard deviation of numerator of :math:`H`.

        Example:

            .. literalinclude:: code/effective_example.py
                :language: python
                :lines: 1-30
                :emphasize-lines: 29

            .. testoutput::

                H_ginibre= -0.055816051215869376

            .. plot:: code/effective_example.py
                :include-source: False

        """

        line = lambda x, a, b: a + b * x
        (intercept, slope), cov = self._fit(line, k_norm_stop, **kwargs)

        self.fitted_line = lambda x: intercept + slope * x

        # Find first peak in structure factor (sf)
        s0 = intercept
        s0_std = np.sqrt(cov[0, 0])

        s_first_peak = 1
        idx_peaks, _ = find_peaks(self.sf, height=s_first_peak)
        if idx_peaks.size:
            self.i_first_peak = max(idx_peaks[0], 1)
            s_first_peak = self.sf[self.i_first_peak]

        return s0 / s_first_peak, s0_std

    # ? how about hyperuniformity_class
    # todo rename k_norm_stop to k_norm_max
    def power_decay(self, k_norm_stop=1, **kwargs):
        r"""Fit a polynomial of the form :math:`y = c \cdot x^{\alpha}`, where `\alpha` characterizes the class of hyperuniformity of the underlying point process, see :cite:`Cos21`.

        Args:
            k_norm_stop (float, optional): the bound on ``k_norm``. Defaults to None.

        Keyword args:
            kwargs (dict): keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            The power decay of the structure factor and the associated approximated :math:`S(0)`.

        Example:

            .. literalinclude:: code/power_decay_example.py
                :language: python
                :emphasize-lines: 29

            .. testoutput::

                The estimated power of the decay to zero of the approximated structure factor is: 1.93893628269006

        """
        poly = lambda x, alpha, c: c * x ** alpha
        (alpha, c), _ = self._fit(poly, k_norm_stop, **kwargs)
        return alpha, c

    def _fit(self, function, x_max, **kwargs):
        """Fit ``function`` using `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.

        #todo finish the docstring

        Args:
            function (callable): function to fit.
            x_max (float): maximum value above

        Keyword args:
            kwargs (dict): keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            tuple: see ouput of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.
        """
        #! todo make sure k_norm, sf and std are sorted according to k_norm values before searching for the index of k_norm whose value is the closest to x_max ...
        #! ... in __init__ preferably
        k_norm = self.k_norm
        sf = self.sf
        std = self.std_sf

        i = len(k_norm)
        if x_max is not None:
            # index of the closest value to x_max in k_norm
            i = np.argmin(np.abs(k_norm.ravel() - x_max))

        xdata = k_norm[:i]
        ydata = sf[:i]
        sigma = std[:i] if std is not None else None
        kwargs["sigma"] = sigma

        return curve_fit(f=function, xdata=xdata, ydata=ydata, **kwargs)
