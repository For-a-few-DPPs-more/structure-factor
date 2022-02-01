"""Collection of functions that compute tapered spectral estimates of the structure factor :math:`S(k)` of a stationary point process given one realization.

Some tapers (tapering function) are available in :ref:`tapers`.
"""

import numpy as np


def tapered_dft(k, point_pattern, taper):
    r"""Compute the tapered discrete Fourier transform (tapered DFT) associated with ``point_pattern`` evaluated at ``k``, using ``taper`` :math:`t`

    .. math::

        \sum_{j=1}^N t(x_j, W) \exp(- i \langle k, x_j \rangle),

    where :math:`x_{1}, \dots, x_{N}` corresponds to ``point_pattern.points`` and :math:`W` corresponds to the window ``point_pattern.window``.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the tapered DFT is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        taper (object): class with static method or instance with method ``.taper(x, window)`` corresponding to :math:`t(x, W)` such that :math:`\|t(\cdot, W)\|_2 = 1`.

    Returns:
        numpy.ndarray: Evaluation of the DFT of the taper h
    """
    points = point_pattern.points
    window = point_pattern.window
    K = np.atleast_2d(k)
    X = np.atleast_2d(points)
    nb_k, _ = K.shape
    nb_x, _ = X.shape

    # dft = sum_x t(x) exp(- i <k, x>)
    hx_exp_ikx = np.zeros((nb_k, nb_x), dtype=complex)
    # i <k, x>
    hx_exp_ikx.imag = np.dot(K, X.T)
    # - i <k, x>
    np.conj(hx_exp_ikx, out=hx_exp_ikx)
    # exp(- i <k, x>)
    np.exp(hx_exp_ikx, out=hx_exp_ikx)
    # t(x) exp(- i <k, x>)
    hx = taper.taper(X, window)
    hx_exp_ikx *= hx
    # sum_x t(x) exp(- i <k, x>)
    dft = np.sum(hx_exp_ikx, axis=1)
    return dft


def periodogram_from_dft(dft):
    r"""Compute the square absolute value ``abs(dft)**2`` of the discrete Fourier transform ``dft``.

    .. math::

        \left\lvert
        \sum_{j=1}^N t(x_j) \exp(- i \langle k, x_j \rangle)
        \right\rvert ^ 2,

    Args:
        dft (numpy.ndarray): discrete Fourier transform computed via :py:func:`~structure_factor.spectral_estimators.tapered_dft`.

    Returns:
        numpy.ndarray: ``abs(dft)**2``.
    """
    periodogram = np.zeros_like(dft, dtype=float)
    np.abs(dft, out=periodogram)
    np.square(periodogram, out=periodogram)
    return periodogram


def tapered_spectral_estimator_core(k, point_pattern, taper):
    r"""Compute the tapered spectral estimator :math:`S_{TP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        taper (object): class with static method or instance with method ``.taper(x, window)`` corresponding to :math:`t(x, W)` such that :math:`\|t(\cdot, W)\|_2 = 1`.

    Returns:
        numpy.ndarray: Evaluation of the tapered spectral estimator :math:`S_{TP}(t, k)` at ``k``.

    .. proof:definition::

        The tapered spectral estimator :math:`S_{TP}(t, k)`, constructed from a realization :math:`\{x_1, \dots, x_N\} \subset \mathbb{R}^d`, is defined by,

        .. math::

            S_{TP}(t, k) =
            \frac{1}{\rho}
            \left\lvert
                \sum_{j=1}^N
                t(x_j, W)
                \exp(- i \left\langle k, x_j \right\rangle)
            \right\rvert^2,

        for :math:`k \in \mathbb{R}^d`.
    """
    rho = point_pattern.intensity
    dft = tapered_dft(k, point_pattern, taper)
    estimated_sf_k = periodogram_from_dft(dft)
    estimated_sf_k /= rho
    return estimated_sf_k


def select_tapered_spectral_estimator(debiased, direct):
    """Select the tapered spectral estimator of the structure factor.

    Args:
        debiased (bool): trigger the use of a debiased tapered estimator.
        direct (bool): If ``debiased`` is True, trigger the use of the direct/undirect debiased tapered estimator.

    Returns:
        callable: According to ``debiased`` and ``direct``

            - :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_debiased_direct`
            - :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_debiased_undirect`
            - :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_core`
    """
    if debiased:
        if direct:
            return tapered_spectral_estimator_debiased_direct
        return tapered_spectral_estimator_debiased_undirect
    return tapered_spectral_estimator_core


# ? renamne tapered_periodogram_debiased_direct and the other functions?
def tapered_spectral_estimator_debiased_direct(k, point_pattern, taper):
    r"""Compute the direct debiased tapered spectral estimator :math:`S_{DDTP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

    .. math::

        S_{DDTP}(t, k) =
            \frac{1}{\rho} \left\lvert
            \sum_{j=1}^N
                t(x_j, W)
                \exp(- i \left\langle k, x_j \right\rangle)
                - \rho F[t(\cdot, W)](k)
            \right\rvert^2

    where :math:`x_{1}, \dots, x_{N}` corresponds to ``point_pattern.points``,  :math:`W` corresponds to the window ``point_pattern.window`` and :math:`\rho` is the intensity of the underlying stationary point process.

    The direct debiased estimator :math:`S_{DDTP}(t, k)` is positive and asymptotically unbiased as the observation window :math:`W` grows to :math:`\mathbb{R}^d`.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        taper (object): class instance with two methods:

            - ``.taper(x, window)`` corresponding to the taper function :math:`t(x, W)` such that :math:`\|t(\cdot, W)\|_2 = 1`.

            - ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the taper function.

    Returns:
        numpy.ndarray: Vector of size :math:`n` containing the evaluation of the direct debiased estimator :math:`S_{DDTP}(t, k)`.
    """
    rho = point_pattern.intensity
    window = point_pattern.window

    # Debiased dft
    dft = tapered_dft(k, point_pattern, taper)
    dft -= rho * taper.ft_taper(k, window)

    estimated_sf_k = periodogram_from_dft(dft)
    estimated_sf_k /= rho
    return estimated_sf_k


def tapered_spectral_estimator_debiased_undirect(k, point_pattern, taper):
    r"""Compute the undirect debiased tapered spectral estimator :math:`S_{UDTP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

    .. math::

        S_{UDTP}(t, k) =
            \frac{1}{\rho}
            \left\lvert
            \sum_{j=1}^N
                t(x_j, W)
                \exp(- i \left\langle k, x_j \right\rangle)
            \right\rvert^2
            - \rho
            \left\lvert
            F[t(\cdot, W)](k))
            \right\rvert^2

    where :math:`x_{1}, \dots, x_{N}` corresponds to ``point_pattern.points``, :math:`W` corresponds to the window ``point_pattern.window`` and :math:`\rho` is the intensity of the underlying stationary point process.

    The undirect debiased estimator :math:`S_{UDTP}(t, k)` is not guaranteed to be positive but is asymptotically unbiased as the observation window :math:`W` grows to :math:`\mathbb{R}^d`.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        taper (object): class instance with two methods:

            - ``.taper(x, window)`` corresponding to the taper function :math:`t(x, W)` such that :math:`\|t(\cdot, W)\|_2 = 1`.
            - ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the taper function.

    Returns:
        numpy.ndarray: Vector of size :math:`n` containing the evaluation of the direct debiased estimator :math:`S_{UDTP}(t, k)`.
    """
    window = point_pattern.window
    rho = point_pattern.intensity

    estimated_sf_k = tapered_spectral_estimator_core(k, point_pattern, taper)
    Hk_2 = np.abs(taper.ft_taper(k, window))
    np.square(Hk_2, out=Hk_2)
    estimated_sf_k -= rho * Hk_2
    return estimated_sf_k


#! add test
def multitapered_spectral_estimator(
    k, point_pattern, *tapers, debiased=False, direct=True
):
    r"""Compute the multitaped spectral estimator :math:`S_{MTP}((t_i)_{i=1}^p, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``tapers`` :math:`(t_i)_{i=1}^p` and the realization ``point_pattern`` of the underlying stationary point process.

    .. math::

        S_{MTP}(k) =
            \frac{1}{p}
            \sum_{i=1}^p
                S_{TP}(t_i, k)

    where :math:`x_{1}, \dots, x_{N}` corresponds to ``point_pattern.points``, :math:`W` corresponds to the window ``point_pattern.window`` and :math:`\rho` is the intensity of the underlying stationary point process.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        tapers (iterable): collection of object instances with two methods:

            - ``.taper(x, window)`` corresponding to the taper function :math:`t(x, W)` such that :math:`\|t(\cdot, W)\|_2 = 1`.

            - ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the taper function, if ``debiased`` is True.

        debiased (bool): trigger the use of debiased tapered spectral estimators.

        direct (bool): if ``debiased`` is True, trigger the use of the tapered direct/undirect debiased spectral estimators.

    Returns:
        numpy.ndarray: Vector of size :math:`n` containing the evaluation of the direct debiased estimator :math:`S_{TP}(t, k)`.

    .. seealso::

        - :py:func:`~structure_factor.spectral_estimators.select_tapered_spectral_estimator`.
    """
    estimator = select_tapered_spectral_estimator(debiased, direct)
    estimated_sf_k = np.zeros(k.shape[0], dtype=float)
    for taper in tapers:
        estimated_sf_k += estimator(k, point_pattern, taper)
    estimated_sf_k /= len(tapers)
    return estimated_sf_k
