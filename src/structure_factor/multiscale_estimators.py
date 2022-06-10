import warnings
import numpy as np
from structure_factor.structure_factor import StructureFactor
from structure_factor.spatial_windows import (
    BoxWindow,
    BallWindow,
    subwindow_parameter_max,
)
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)
from scipy.stats import poisson


def multiscale_estimator(
    point_pattern,
    estimator,
    subwindows_list,
    k_list,
    mean_poisson,
    m=None,
    proba_list=None,
    verbose=True,
    **kwargs,
):
    r"""Sample from :math:`Z`  :cite:`HGBLR:22` using a PointPattern and a realization from the r.v. :math:`M`. See the definition of :math:`Z` below.

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): An encapsulation of a realization of a point process, the observation window, and (optionally) the intensity of the point process.

        estimator (str): Choice of structure factor's estimator. The parameters of the chosen estimator must be added as keyword arguments. The available estimators are "scattering_intensity", "tapered_estimator", "bartlett_isotropic_estimator", and "quadrature_estimator_isotropic". See :py:class:`~structure_factor.structure_factor.StructureFactor`.

        subwindows_list (list): List of increasing cubic or ball-shaped :py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, typically, obtained using :py:func:`~structure_factor.multiscale_estimators.subwindows_list`. The shape of the windows depends on the choice of the ``estimator``. Each element of ``point_pattern_list`` will be restricted to these windows to compute :math:`Z`.

        k_list (list): List of wavevectors (or wavenumbers) where the ``estimator`` is to be evaluated. Each element is associated with an element of ``subwindows_list``. Typically, obtained using :py:func:`~structure_factor.multiscale_estimators.subwindows_list`.

        mean_poisson (int): Parameter of the Poisson r.v. :math:`M` used to compute :math:`Z`. To use a different distribution of the r.v. :math:`M`, set ``mean_poisson=None`` and specify ``m_list`` and ``proba_list`` corresponding to :math:`M`.

        m (int, optional): Realization of the positive integer-valued r.v. :math:`M` used when ``mean_poisson=None``. Defaults to None.

        proba_list (list, optional): List of :math:`\mathbb{P}(M \geq j)` used  with ``m`` when ``mean_poisson=None``. Should contains at least ``m`` elements. Defaults to None.

        verbose (bool, optional): If "True" and ``mean_poisson`` is not None, print the re-sampled values of :math:`M`. Defaults to False.

    Keyword Args:
        kwargs (dict): Parameters of the chosen ``estimator`` of the structure factor.  See :py:class:`~structure_factor.structure_factor.StructureFactor`.

    Returns:
        float: The obtained value of :math:`Z`.

    Example:
        .. literalinclude:: code/multiscale_estimators/multiscale_estimator.py
            :language: python

    .. proof:definition::

        Let :math:`\mathcal{X} \in \mathbb{R}^d` be a stationary point process of which we consider an increasing sequence of sets :math:`(\mathcal{X} \cap W_m)_{m \geq 1}`, with :math:`(W_m)_m` centered box (or ball)-shaped windows s.t. :math:`W_s \subset W_r` for all :math:`0< s<r`, and :math:`W_{\infty} = \mathbb{R}^d`.
        We define the sequence of r.v. :math:`Y_m = 1\wedge \widehat{S}_m(\mathbf{k}_m^{\text{min}})`, where :math:`\widehat{S}_m` is one of the positive, asymptotically unbiased estimators of the structure factor of :math:`\mathcal{X}` applied on the observation :math:`\mathcal{X} \cap W_m`, and :math:`\mathbf{k}_m^{\text{min}}` is the minimum allowed wavevector associated with :math:`W_m`.
        Then, :math:`Z` is defined by,

        .. math::

            Z = \sum_{j=1}^{M} \frac{Y_j - Y_{j-1}}{\mathbb{P}(M\geq j)}

        with :math:`M` an :math:`\mathbb{N}`-valued random variable such that :math:`\mathbb{P}(M \geq j)>0` for all :math:`j`, and :math:`Y_{0}=0` :cite:`HGBLR:22`, :cite:`RhGl15`.



    """

    # r.v. threshold
    m_thresh = m_threshold(
        window_min=subwindows_list[0],
        window_max=subwindows_list[-1],
    )
    # r.v. M
    if m is None:
        m = _poisson_rv(mean_poisson, m_thresh, verbose=verbose)
    else:
        if m > m_thresh:
            warnings.warn(
                message=f"The random variable M exceed the allowed threshold {m_thresh}."
            )
    m = int(m)
    # proba list
    if proba_list is None:
        proba_list = 1 - (
            poisson.cdf(k=range(1, m + 1), mu=mean_poisson)
            - poisson.pmf(k=range(1, m + 1), mu=mean_poisson)
        )
    else:
        if len(proba_list) < m:
            raise ValueError(f"The proba list should contains at least{m} elements.")
        proba_list = proba_list[:m]

    # k and subwindows list
    if len(subwindows_list) != len(k_list):
        raise ValueError(
            "The number of wavevectors/wavenumber (k) should be equal to the number of subwindows. Each k is associated to a subwindow."
        )
    if len(subwindows_list) < m:
        raise ValueError(
            f"The number of subwindows {len(subwindows_list)} should be at least equal to the random variable M= {m}."
        )
    subwindows_list = subwindows_list[:m]
    k_list = k_list[:m]

    # approximated s_k_min list
    s_k_min_list = multiscale_estimator_core(
        point_pattern=point_pattern,
        subwindows_list=subwindows_list,
        k_list=k_list,
        estimator=estimator,
        **kwargs,
    )
    # the r.v. (y_n)_n
    y_list = [min(np.array([1]), s) for s in s_k_min_list]
    z = coupled_sum_estimator(y_list, proba_list)

    return z


def multiscale_estimator_core(
    point_pattern, subwindows_list, k_list, estimator, **kwargs
):
    point_pattern_list = [point_pattern.restrict_to_window(w) for w in subwindows_list]
    estimated_sf_k_list = [
        _select_structure_factor_estimator(
            point_pattern=p, estimator=estimator, k=q, **kwargs
        )
        for p, q in zip(point_pattern_list, k_list)
    ]
    return estimated_sf_k_list


def coupled_sum_estimator(y_list, proba_list):
    r"""The coupled sum estimator of :cite:`RhGl15`.

    Args:
        y_list (list): List of :math:`M` realizations of the r.v. :math:`Y`.
        proba_list (list): List of :math:`\mathbb{P}(M \geq j)` with :math:`1 \leq j \leq M`.

    Returns:
        float: Obtained value of the coupled sum estimator.

    Example:
        .. literalinclude:: code/multiscale_estimators/coupled_sum_estimator.py
            :language: python

    .. proof:definition::

        Let :math:`(Y_m)_{m\geq 1}` be a sequence of :math:`L^2` approximations of a r.v. :math:`Y` each of which can be generated in finite time, for which :math:`\mathbb{E}^{1/2}[(Y_m - Y)^2]` converges to zero as :math:`m` goes to infinity.
        The coupled sum estimator of :cite:`RhGl15` is defined by,

        .. math::

            Z = \sum_{j=1}^{M} \frac{Y_j - Y_{j-1}}{\mathbb{P}(M\geq j)},

        with :math:`M` an :math:`\mathbb{N}`-valued random variable such that :math:`\mathbb{P}(M \geq j)>0` for all :math:`j`, and :math:`Y_{0}=0`.

    """
    y_list_with_0 = np.append(0, y_list)  # 0 first element of the list
    y_pairwise_diff = np.array(
        [t - s for s, t in zip(y_list_with_0[:-1], y_list_with_0[1:])]
    )
    y_pairwise_diff = y_pairwise_diff / np.array(proba_list)
    z = np.sum(y_pairwise_diff)
    return z


def subwindows_list(
    window, subwindows_type="BoxWindow", param_0=None, param_max=None, params=None
):
    """Create a list of cubic (or ball)-shaped subwindows of a father window, with the associated minimum allowed wavevectors (or wavenumbers).

    Args:
        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Father window.

        subwindows_type (str, optional): Type of the subwindows to be created. The available types are "BoxWindow" and "BallWindow". The former for cubic and the latter for ball-shaped subwindows. Defaults to "BoxWindow".

        param_0 (float, optional): Parameter (lengthside/radius) of the first subwindow to be created. If not None, an increasing sequence of subwindows with parameters of unit increments is created. The biggest subwindow has parameter ``param_max`` if it's not None, else, the maximum possible parameter. Defaults to None.

        param_max (float, optional): Maximum subwindow parameter (lengthside/radius). Used when ``param_0`` is not None. Defaults to None.

        params (list, optional): List of parameters (lengthside/radius) of the output subwindows. For a list of parameters of unit increments, ``param_0`` and ``param_max`` can be used instead. Defaults to None.

    Returns:
        (list, list):
            - subwindows: Obtained subwindows.
            - k: Minimum allowed wavevectors of :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity` or wavenumbers of :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm_bartlett_isotropic` associated with the subwindow list. The former is for cubic and the latter for ball-shaped subwindows.

    Example:
        .. plot:: code/multiscale_estimators/subwindows.py
            :include-source: True

    .. seealso::

        - :py:mod:`~structure_factor.spatial_windows`
        - :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity`
        - :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm_bartlett_isotropic`

    """
    d = window.dimension
    # parameter of the biggest possible subwindow contained in `window``
    window_param_max = subwindow_parameter_max(window, subwindows_type)
    # subwindows list of parameters
    if params is None:
        if param_0 is None:
            raise ValueError(
                "The minimum window parameter is mandatory. Hint: specify the minimum window parameter."
            )
        # from param_0 till param_max with unit space
        if param_max is None:
            params = np.arange(param_0, window_param_max)
    else:
        if max(params) > window_param_max:
            raise ValueError(
                f"The maximum sub-window (parameter={max(params)}) is bigger than the father window (parameter={window_param_max}). Hint: Reduce the maximum subwindow parameter. "
            )
    # subwindows and the associated k
    if subwindows_type == "BallWindow":
        subwindows = [BallWindow(center=[0] * d, radius=r) for r in params]
        # check if d is even
        _, mod_ = divmod(d, 2)
        is_even_dimension = mod_ == 0
        if is_even_dimension:
            k_list = [
                allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=1)
                for r in params
            ]
        else:
            k_list = None
            warnings.warn(
                message=f"Isotropic allowed wavenumber are available only when the dimension of the space is an even number (i.e., d/2 is an integer)."
            )

    else:
        subwindows = [BoxWindow(bounds=[[-l / 2, l / 2]] * d) for l in params]
        k_list = [np.full((1, d), fill_value=2 * np.pi / l) for l in params]
    return subwindows, k_list


def m_threshold(window_min, window_max):
    r"""Find the maximum number of integers ranging between the parameters (lengthside/radius) of ``window_min`` and the largest subwindow of ``window_max`` having the shape of ``window_min``.
    In particular, it gives the maximum value of the r.v. :math:`M` that can be used to compute the :py:func:`~structure_factor.multiscale_estimators.multiscale_estimator` given the smallest and biggest subwindow.

    Args:
        window_min (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Smallest cubic or ball-shaped window centered at the origin.
        window_max (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Biggest box or ball-shaped window centered at the origin.

    Returns:
        int : Maximum number of integers ranging between the parameters (lengthside/radius) of ``window_min`` and the largest subwindow of ``window_max`` having the shape of ``window_min``.

    Example:
        .. plot:: code/multiscale_estimators/m_threshold.py
            :include-source: True

    .. seealso::

        - :py:mod:`~structure_factor.spatial_windows`
    """
    if isinstance(window_min, BoxWindow):
        subwindow_type = "BoxWindow"
    else:
        subwindow_type = "BallWindow"
    param_max = subwindow_parameter_max(window_max, subwindow_type=subwindow_type)
    param_min = subwindow_parameter_max(window_min, subwindow_type=subwindow_type)
    if param_max < param_min:
        raise ValueError("window_min should be bigger than window_max.")
    m_threshold = int(param_max - param_min)
    return m_threshold


def _select_structure_factor_estimator(point_pattern, estimator, k, **kwargs):
    sf = StructureFactor(point_pattern)
    if estimator == "scattering_intensity":
        _, estimated_sf_k = sf.scattering_intensity(k=k, **kwargs)
    elif estimator == "tapered_estimator":
        _, estimated_sf_k = sf.tapered_estimator(k=k, **kwargs)
    elif estimator == "bartlett_isotropic_estimator":
        _, estimated_sf_k = sf.bartlett_isotropic_estimator(k_norm=k, **kwargs)
    elif estimator == "quadrature_estimator_isotropic":
        _, estimated_sf_k = sf.quadrature_estimator_isotropic(k_norm=k, **kwargs)
    else:
        raise ValueError(
            "Available estimators: 'scattering_intensity', 'tapered_estimator', 'bartlett_isotropic_estimator', 'quadrature_estimator_isotropic'. "
        )
    return estimated_sf_k


def _poisson_rv(mean_poisson, threshold, verbose=True):

    m = poisson.rvs(mu=mean_poisson, size=1)
    while m > threshold:
        if verbose:
            print("Re-sample M; current M= ", m, ", threshold=", threshold)
        m = int(poisson.rvs(mu=mean_poisson, size=1))
    return m
