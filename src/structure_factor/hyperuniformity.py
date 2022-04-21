"""Class :py:class:`~structure_factor.hyperuniformity.HyperUniformity`  designed to study the hyperuniformity of a stationary point process given an estimation of its structure factor :py:class:`~structure_factor.structure_factor.StructureFactor`.

- :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.bin_data`: Method for regularizing the structure factor estimation.
- :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.effective_hyperuniformity`: Test of effective hyperuniformity.
- :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.hyperuniformity_class`: Test of the possible class of hyperuniformity.

.. note::

    **Typical usage**

    1. Estimate the structure factor of the point process by one of the methods of :py:class:`~structure_factor.structure_factor.StructureFactor`.

    2. Regularize the results using :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.bin_data`, if needed.

    3. Test the effective hyperuniformity using :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.effective_hyperuniformity`.

    4. If the results of :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.effective_hyperuniformity` approve the effective hyperuniformity hypothesis, find the possible class of hyperuniformity using :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.hyperuniformity_class`.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from structure_factor.utils import _bin_statistics, _sort_vectors


class Hyperuniformity:
    r"""Class collecting some diagonistics of hyperuniformity for stationary isotropic (or effectively isotropic) point processes :math:`\mathcal{X} \subset \mathbb{R}^d`, given an estimation of the structure factor.

    .. todo::

        list attributes

    .. proof:definition::

        A stationary point process :math:`\mathcal{X}` is said to be hyperuniform if its structure factor :math:`S` vanishes at 0.
        For more details, we refer to :cite:`HGBLR:22`, (Section 2).
    """

    def __init__(self, k_norm=None, sf=None, std_sf=None, sf_k_min_list=None):
        """Initialize the object from the pair ``k_norm, sf`` which corresponds to the norm of the wavevector (denoted wavenumber) and the evaluation of the structure factor.

        Args:
            k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e. norms of the wavevectors). Default to None.

            sf (numpy.ndarray, optional): Vector of evaluations of the structure factor, of the given point process, at :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. Default to None.

            std (numpy.ndarray, optional): Vector of standard deviations associated to :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`. Defaults to None.

            sf_k_min_list (list, optional): List of estimations of the structure factor of the given point process using one realization restricted to an increasing set of observation windows. Each element of the list depends on the approximation of the structure factor at the minimum wavevector associated with the corresponding observation window. Typically we consider the minimums between the estimations and one. Default to None.

        """
        if k_norm is not None and sf is not None:
            assert isinstance(k_norm, np.ndarray)
            assert isinstance(sf, np.ndarray)
            assert k_norm.shape == sf.shape
            # k_norm sorted
            self.k_norm, self.sf, self.std_sf = _sort_vectors(k_norm, sf, std_sf)

        self.sf_k_min_list = sf_k_min_list
        self.fitted_line = None  # fitted line to sf near zero
        self.i_first_peak = None  # index of first peak of sf
        self.fitted_poly = None  # fitted polynomial to sf near zero

    def bin_data(self, **params):
        """Split the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm` into sub-intervals (or bins) and evaluate, over each sub-interval, the mean and the standard deviation of the corresponding values in the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`.

        Keyword Args:
            params (dict): Keyword arguments (except ``"x"``, ``"values"`` and ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray):
                - ``k_norm``: Centers of the bins (update the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`).
                - ``sf``: Means of the structure factor over the bins (update the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`).
                - ``std_sf``: Standard deviations of the structure factor over the bins (update the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.std_sf`).

        Example:
            .. plot:: code/hyperuniformity/bin_data.py
                :include-source: True

        .. seealso::

            - :py:class:`~structure_factor.structure_factor.StructureFactor`
        """

        if self.sf is None or self.k_norm is None:
            raise ValueError(
                "The attribute `sf` and `k_norm` of the class `Hyperuniformity` are mandatory for this method. Hint: re-initialize the class Hyperuniformity and update `k_norm` and `sf`."
            )
        self.k_norm, self.sf, self.std_sf = _bin_statistics(
            self.k_norm, self.sf, **params
        )
        return self.k_norm, self.sf, self.std_sf

    def effective_hyperuniformity(self, k_norm_stop, **kwargs):
        r"""Evaluate the index :math:`H` of hyperuniformity of a point process using its structure factor :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`. If :math:`H<10^{-3}` the corresponding point process is deemed effectively hyperuniform.

        Args:
            k_norm_stop (float): Threshold on :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. Used to find the numerator of :math:`H` by linear regression of :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` up to the value associated with ``k_norm_stop``.

        Keyword Args:
            kwargs (dict): Keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            tuple(float, float):
                - H: Value of the index :math:`H`.
                - s0_std: Standard deviation of the numerator of :math:`H`.

        Example:
            .. plot:: code/hyperuniformity/effective_hyperuniformity.py
                :include-source: True

        .. proof:definition::

            A stationary isotropic point process :math:`\mathcal{X} \subset \mathbb{R}^d`, is said to be effectively hyperuniform if :math:`H \leq 10^{-3}` where :math:`H` is defined following :cite:`Tor18` (Section 11.1.6) and :cite:`KlaAl19` (supplementary Section 8) by,

            .. math::

                H = \frac{\hat{S}(\mathbf{0})}{S(\mathbf{k}_{peak})}\cdot

            - :math:`S` is the structure factor of :math:`\mathcal{X}`,
            - :math:`\hat{S}(\mathbf{0})` is a linear extrapolation of the structure factor at :math:`\mathbf{k}=\mathbf{0}`,
            - :math:`\mathbf{k}_{peak}` is the location of the first dominant peak value of :math:`S`.

            For more details, we refer to :cite:`HGBLR:22` (Section 2.5).

        .. important::

            To compute the numerator :math:`\hat{S}(\mathbf{0})` of :math:`H`, a line is fitted using a linear extrapolation with a least-square fit on the values of :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` associated to the sub-vector of :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm` truncated around the threshold ``k_norm_stop``. ``k_norm_stop`` must satisfy a good compromise of being close to zero and allowing to fit the line on a sufficient number of points.

            If the standard deviations of :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` are provided in the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.std_sf` then, these values will be considered while fitting the line.

        .. seealso::

            - :py:class:`~structure_factor.structure_factor.StructureFactor`
            - :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.bin_data`
            - :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.hyperuniformity_class`
        """
        if self.sf is None or self.k_norm is None:
            raise ValueError(
                "The attribute `sf` and `k_norm` of the class `Hyperuniformity` are mandatory for this method. Hint: re-initialize the class Hyperuniformity and update `k_norm` and `sf`."
            )
        line = lambda x, a, b: a + b * x
        (intercept, slope), cov = self._fit(line, k_norm_stop, **kwargs)

        self.fitted_line = lambda x: intercept + slope * x

        # Find first peak in structure factor (sf)
        s0 = intercept
        s0_std = np.sqrt(cov[0, 0])

        s_first_peak = 1
        idx_peaks, _ = find_peaks(self.sf, height=s_first_peak)
        if idx_peaks.size:
            self.i_first_peak = idx_peaks[0]
            s_first_peak = max(self.sf[self.i_first_peak], 1)
        H = s0 / s_first_peak

        return H, s0_std

    #! add docs
    def multiscale_test(self, proba_list):
        sf_k_min_list = self.sf_k_min_list
        if sf_k_min_list is None:
            raise ValueError(
                "The attribute `sf_k_min_list` of the class `Hyperuniformity` is mandatory for this test. Hint: re-initialize the class Hyperuniformity and update `sf_k_min_list`."
            )
        sf_k_min_with_0 = np.append(0, sf_k_min_list)  # 0 first element of the list
        estimation_pairwise_diff = np.array(
            [t - s for s, t in zip(sf_k_min_with_0[:-1], sf_k_min_with_0[1:])]
        )
        estimation_pairwise_diff = estimation_pairwise_diff / np.array(proba_list)
        z = np.sum(estimation_pairwise_diff)
        return z

    def hyperuniformity_class(self, k_norm_stop=1, **kwargs):
        r"""Fit a polynomial :math:`y = c \cdot x^{\alpha}` to the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` around zero. :math:`\alpha` is used to specify the possible class of hyperuniformity of the associated point process (as described below).

        Args:
            k_norm_stop (float, optional): Threshold on :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. A polynomial will be fitted on :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` starting from zero until the value associated with ``k_norm_stop``. ``k_norm_stop`` should be sufficiently close to zero to get a good result. Defaults to 1.

        Keyword Args:
            kwargs (dict): Keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            tuple(float, float):
                - alpha: The estimated power decay of the structure factor.
                - c: The approximated value of the structure factor on the origin.

        Example:
            .. plot:: code/hyperuniformity/hyperuniformity_class.py
                :include-source: True

        .. proof:definition::

            For a stationary  hyperuniform point process :math:`\mathcal{X} \subset \mathbb{R}^d`, if :math:`\vert S(\mathbf{k})\vert\sim c \Vert \mathbf{k} \Vert_2^\alpha` in the neighborhood of 0, then by :cite:`Cos21` (Section 4.1.) the value of :math:`\alpha` determines the hyperuniformity class of :math:`\mathcal{X}` as follows,

            +-------+----------------+---------------------------------------------------------------+
            | Class | :math:`\alpha` | :math:`\mathbb{V}\text{ar}\left[\mathcal{X}(B(0, R)) \right]` |
            +=======+================+===============================================================+
            | I     | :math:`> 1`    | :math:`\mathcal{O}(R^{d-1})`                                  |
            +-------+----------------+---------------------------------------------------------------+
            | II    | :math:`= 1`    | :math:`\mathcal{O}(R^{d-1}\log(R))`                           |
            +-------+----------------+---------------------------------------------------------------+
            | III   | :math:`]0, 1[` | :math:`\mathcal{O}(R^{d-\alpha})`                             |
            +-------+----------------+---------------------------------------------------------------+

            For more details, we refer to :cite:`HGBLR:22`, (Section 2.5).

        .. seealso::

            - :py:class:`~structure_factor.structure_factor.StructureFactor`
            - :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.bin_data`
            - :py:meth:`~structure_factor.hyperuniformity.Hyperuniformity.effective_hyperuniformity`
        """
        if self.sf is None or self.k_norm is None:
            raise ValueError(
                "The attribute `sf` and `k_norm` of the class `Hyperuniformity` are mandatory for this method. Hint: re-initialize the class Hyperuniformity and update `k_norm` and `sf`."
            )
        poly = lambda x, alpha, c: c * x ** alpha
        (alpha, c), _ = self._fit(poly, k_norm_stop, **kwargs)
        self.fitted_poly = lambda x: c * x ** alpha
        return alpha, c

    # todo clarify x_max
    def _fit(self, function, x_max, **kwargs):
        """Fit ``function`` using `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.

        Args:
            function (callable): Function to fit.

            x_max (float): Maximum value above.

        Keyword Args:
            kwargs (dict): Keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

        Returns:
            tuple: See output of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.
        """
        k_norm = self.k_norm
        sf = self.sf
        std = self.std_sf

        i = len(k_norm)
        if x_max is not None:
            # index of the closest value to x_max in k_norm
            i = np.argmin(np.abs(k_norm - x_max))

        xdata = k_norm[:i]
        ydata = sf[:i]
        if std is not None and (std != 0).all():
            sigma = std[:i]
            kwargs["sigma"] = sigma

        return curve_fit(f=function, xdata=xdata, ydata=ydata, **kwargs)
