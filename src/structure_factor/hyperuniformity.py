"""Functions designed to study the hyperuniformity of a stationary point process using estimation(s) of its structure factor :py:class:`~structure_factor.structure_factor.StructureFactor`.

Hyperuniformity diagnostics functions:

    - :py:func:`~structure_factor.hyperuniformity.multiscale_test`: Statistical test of hyperuniformity asymptomatically valid.
    - :py:func:`~structure_factor.hyperuniformity.effective_hyperuniformity`: Test of effective hyperuniformity.
    - :py:func:`~structure_factor.hyperuniformity.hyperuniformity_class`: Estimation of the possible class of hyperuniformity.

Additional functions:
    - :py:func:`~structure_factor.hyperuniformity.bin_data`: Method for regularizing structure factor's estimation.
    - :py:func:`~structure_factor.hyperuniformity.subwindows`: Method for generating a list of subwindows from a father window with the corresponding minimal allowed wavevectors (or wavenumbers).


.. note::

    **Typical usage**

    1. Estimate the structure factor of the point process by one of the methods of :py:class:`~structure_factor.structure_factor.StructureFactor`.

    2. Regularize the results using :py:func:`~structure_factor.hyperuniformity.bin_data`, if needed.

    3. Test the hyperuniformity using the statistical test :py:func:`~structure_factor.hyperuniformity.multiscale_test` or the test of effective hyperuniformity :py:func:`~structure_factor.hyperuniformity.effective_hyperuniformity`.

    4. If the hyperuniformity was approved, find the possible hyperuniformity class using :py:func:`~structure_factor.hyperuniformity.hyperuniformity_class`.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import statistics as stat

from structure_factor.utils import _bin_statistics, _sort_vectors
from structure_factor.multiscale_estimators import (
    multiscale_estimator,
    _poisson_rv,
    m_threshold,
    subwindows_list,
)


def multiscale_test(
    point_pattern_list,
    estimator,
    subwindows_list,
    k_list,
    mean_poisson,
    m_list=None,
    proba_list=None,
    verbose=False,
    **kwargs,
):
    r"""Computes the sample mean and its standard deviation of the coupled sum estimator :math:`Z` of a point process using a list of PointPatterns and realizations from the random variable :math:`M`.

    The test rejects the hyperuniformity hypothesis if the confidence interval of :math:`\mathbb{E}[Z]` doesn't contain zero, and vice-versa.
    (See the multiscale hyperuniformity test in :cite:`HGBLR:22`).

    Args:
        point_pattern_list (list): List of :py:class:`~structure_factor.point_pattern.PointPattern` (s). Each element of the list is an encapsulation of a realization of a point process, the observation window, and (optionally) the intensity of the point process. All :py:class:`~structure_factor.point_pattern.PointPattern` (s) should have the same window and intensity.

        estimator (str): Choice of structure factor's estimator. The parameters of the chosen estimator must be specified as keyword arguments. The available estimators are "scattering_intensity", "tapered_estimator", "bartlett_isotropic_estimator", and "quadrature_estimator_isotropic". See :py:class:`~structure_factor.structure_factor.StructureFactor`.

        subwindows_list (list): List of :py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`. An increasing list of windows corresponding to the ``estimator``. Each element of ``point_pattern_list`` will be restricted to these windows to compute :math:`Z`. Typically, obtained using :py:func:`~structure_factor.hyperuniformity.subwindows`.


        k_list (list): List of wavevectors (or wavenumbers) where the ``estimator`` is to be evaluated. Each element is associated with an element of ``subwindows_list``. Typically, obtained using :py:func:`~structure_factor.hyperuniformity.subwindows`.

        mean_poisson (int): Parameter of the Poisson r.v. :math:`M` used to compute :math:`Z`. To use a different distribution of the r.v. :math:`M`, set ``mean_poisson=None`` and specify ``m_list`` and ``proba_list`` corresponding to :math:`M`.

        m_list (list, optional): List of positive integers realizations of the r.v. :math:`M`, used when ``mean_poisson=None``. Each element of ``m_list`` is associated with an element of ``point_pattern_list`` to compute a realization of :math:`Z`. Defaults to None.

        proba_list (list, optional): List of :math:`\mathbb{P}(M>j)` used  with ``m_list`` when ``mean_poisson=None``. Should contains at least ``max(m_list)`` elements. Defaults to None.

        verbose (bool, optional): If "True", print the re-sampled values of :math:`M` when ``mean_poisson`` is not None. Defaults to False.

    Keyword Args:
        kwargs (dict): Parameters of the chosen ``estimator`` of the structure factor.  See :py:class:`~structure_factor.structure_factor.StructureFactor`.

    Returns:
        dict:
        Dictionary containing:
            - "mean_Z": The sample mean of :math:`Z`.
            - "std_mean_Z": The sample standard deviation of :math:`\mathbb{E}[Z]` i.e. sample standard deviation of :math:`Z` divided by the square root of the number of samples.
            - "Z": The obtained values of :math:`Z`.
            - "M": The used values of :math:`M`.

    Example:
        .. plot:: code/hyperuniformity/multiscale_test.py
            :include-source: True

    .. proof:definition::

        Let :math:`\mathcal{X} \in \mathbb{R}^d` be a stationary point process of which we consider an increasing sequence of sets :math:`(\mathcal{X} \cap W_m)_{m \geq 1}`, with :math:`(W_m)_m` centered box (or ball)-shaped windows s.t. :math:`W_s \subset W_r` for all :math:`0< s<r`, and :math:`W_{\infty} = \mathbb{R}^d`.
        We define the sequence of r.v. :math:`Y_m = 1\wedge \widehat{S}_m(\mathbf{k}_m^{\text{min}})`, where :math:`\widehat{S}_m` is one of the positive, asymptotically unbiased estimators of the structure factor of :math:`\mathcal{X}` applied on the observation :math:`\mathcal{X} \cap W_m`, and :math:`\mathbf{k}_m^{\text{min}}` is the minimum allowed wavevector associated with :math:`W_m`.

        Under some assumptions (:cite:`HGBLR:22`, Section 4) :math:`\mathcal{X}` is hyperuniform iff :math:`\mathbb{E}[Z]=0`.
        Where :math:`Z` is the coupled sum estimator defined by,

        .. math::

            \sum_{j=0}^{M} \frac{Y_j - Y_{j-1}}{\mathbb{P}(M\geq j)}

        with :math:`M` an :math:`\mathbb{N}` -valued random variable such that :math:`\mathbb{P}(M \geq j)>0` for all :math:`j`, and :math:`Y_{0}=0`.

    .. note::

            **Typical usage**

            - The function :py:func:`~structure_factor.hyperuniformity.subwindows` can be used to generate from a father window a list of subwindows with the associated allowed wavevectors/wavenumbers.

    .. seealso::

        - :py:meth:`~structure_factor.point_pattern.PointPattern`
        - :py:class:`~structure_factor.structure_factor.StructureFactor`
        - :py:func:`~structure_factor.hyperuniformity.subwindows`
    """
    nb_sample = len(point_pattern_list)
    if m_list is None:
        m_thresh = m_threshold(
            window_min=subwindows_list[0], window_max=subwindows_list[-1]
        )
        m_list = [
            _poisson_rv(mean_poisson, threshold=m_thresh, verbose=verbose)
            for _ in range(nb_sample)
        ]
    z_list = [
        multiscale_estimator(
            p,
            estimator,
            k_list,
            subwindows_list,
            mean_poisson=mean_poisson,
            m=m,
            proba_list=proba_list,
            verbose=verbose,
            **kwargs,
        )
        for (p, m) in zip(point_pattern_list, m_list)
    ]
    mean_z = stat.mean(z_list)
    std_mean_z = stat.stdev(z_list) / np.sqrt(nb_sample)
    summary = {"mean_Z": mean_z, "std_mean_Z": std_mean_z, "Z": z_list, "M": m_list}
    return summary


def effective_hyperuniformity(k_norm, sf, k_norm_stop, std_sf=None, **kwargs):
    r"""Evaluate the index :math:`H` of hyperuniformity of a point process using its structure factor :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`. If :math:`H<10^{-3}` the corresponding point process is deemed effectively hyperuniform.

    Args:
        k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e. norms of the wavevectors). Default to None.
        sf (numpy.ndarray, optional): Vector of evaluations of the structure factor, of the given point process, at :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. Default to None.
        std_sf (numpy.ndarray, optional): Vector of standard deviations associated to :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`. Defaults to None.
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
    # sort vectors
    k_norm, sf, std_sf = _sort_vectors(k_norm, sf, std_sf)
    line = lambda x, a, b: a + b * x
    (intercept, slope), cov = _fit(k_norm, sf, std_sf, line, k_norm_stop, **kwargs)

    fitted_line = lambda x: intercept + slope * x

    # Find first peak in structure factor (sf)
    s0 = intercept
    s0_std = np.sqrt(cov[0, 0])

    s_first_peak = 1
    idx_peaks, _ = find_peaks(sf, height=s_first_peak)
    if idx_peaks.size:
        idx_first_peak = idx_peaks[0]
        s_first_peak = max(sf[idx_first_peak], 1)
    else:
        idx_first_peak = None
    H = s0 / s_first_peak
    summary = {
        "H": H,
        "s0_std": s0_std,
        "fitted_line": fitted_line,
        "idx_first_peak": idx_first_peak,
    }
    return summary


# todo update doc and test
def hyperuniformity_class(k_norm, sf, k_norm_stop=1, std_sf=None, **kwargs):
    r"""Fit a polynomial :math:`y = c \cdot x^{\alpha}` to the attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf` around zero. :math:`\alpha` is used to specify the possible class of hyperuniformity of the associated point process (as described below).

    Args:
        k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e. norms of the wavevectors). Default to None.
        sf (numpy.ndarray, optional): Vector of evaluations of the structure factor, of the given point process, at :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. Default to None.
        std (numpy.ndarray, optional): Vector of standard deviations associated to :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`. Defaults to None.
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
    k_norm, sf, std_sf = _sort_vectors(k_norm, sf, std_sf)
    poly = lambda x, alpha, c: c * x ** alpha
    (alpha, c), _ = _fit(k_norm, sf, std_sf, poly, k_norm_stop, **kwargs)
    fitted_poly = lambda x: c * x ** alpha
    summary = {"alpha": alpha, "c": c, "fitted_poly": fitted_poly}
    return summary


# todo update doc and test
def bin_data(k_norm, sf, **params):
    """Split the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm` into sub-intervals (or bins) and evaluate, over each sub-interval, the mean and the standard deviation of the corresponding values in the vector attribute :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.sf`.

    Args:
        k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e. norms of the wavevectors). Default to None.

        sf (numpy.ndarray, optional): Vector of evaluations of the structure factor, of the given point process, at :py:attr:`~structure_factor.hyperuniformity.Hyperuniformity.k_norm`. Default to None.
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
    k_norm, sf, _ = _sort_vectors(k_norm, sf)
    return _bin_statistics(k_norm, sf, **params)


# todo update doc
def subwindows(
    window, subwindows_type="BoxWindow", param_0=None, param_max=None, params=None
):
    """_summary_

    _extended_summary_

    Args:
        window (_type_): _description_
        subwindows_type (str, optional): _description_. Defaults to "BoxWindow".
        param_0 (_type_, optional): _description_. Defaults to None.
        param_max (_type_, optional): _description_. Defaults to None.
        params (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    return subwindows_list(window, subwindows_type, param_0, param_max, params)


# todo doc
# todo clarify x_max
def _fit(x, y, std, function, x_max, **kwargs):
    """Fit ``function`` using `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.

    Args:
        function (callable): Function to fit.

        x_max (float): Maximum value above.

    Keyword Args:
        kwargs (dict): Keyword arguments (except ``"sigma"``) of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_ parameters.

    Returns:
        tuple: See output of `scipy.scipy.optimize.curve_fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html>`_.
    """

    i = len(x)
    if x_max is not None:
        # index of the closest value to x_max in k_norm
        i = np.argmin(np.abs(x - x_max))

    xdata = x[:i]
    ydata = y[:i]
    if std is not None and (std != 0).all():
        sigma = std[:i]
        kwargs["sigma"] = sigma

    return curve_fit(f=function, xdata=xdata, ydata=ydata, **kwargs)
