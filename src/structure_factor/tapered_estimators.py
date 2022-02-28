"""Collection of functions that compute tapered estimates of the structure factor :math:`S(k)` of a stationary point process given one realization.

Some tapers (tapering function) are available in :ref:`tapers`.
"""

import warnings

import numpy as np

from structure_factor.tapers import BartlettTaper
from structure_factor.utils import meshgrid_to_column_matrix


def scattering_intensity(k, point_pattern, debiased=False, direct=True):
    r"""Compute the scattering intensity estimator :math:`\widehat{S}_{\mathrm{SI}}` of the structure factor evaluated at at ``k``, from one realization of a stationary point process encapsulated in ``point_pattern``.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        debiased (bool, optional): trigger the use of debiased tapered estimators. Defaults to False.

        direct (bool, optional): if ``debiased`` is True, trigger the use of the tapered direct/undirect debiased tapered estimators. Defaults to True.

    Returns:
        numpy.ndarray: Vector of size :math:`n` containing the evaluation of scattering intensity estimator :math:`\widehat{S}_{\mathrm{SI}}(k)`.

    .. proof:definition::

        The scattering intensity estimator :math:`\widehat{S}_{\mathrm{SI}}, constructed from a realization :math:`\{x_1, \dots, x_N\} \subset \mathbb{R}^d`, is defined by,

        .. math::
            \widehat{S}_{\mathrm{SI}}(\mathbf{k}) =
                \frac{1}{N}\left\lvert
                    \sum_{j=1}^N
                        \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                \right\rvert^2 .

        For more details we refer to :cite:`HGBLR:22`, (Section 3.1).

    .. seealso::

        - :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity`
    """
    n, d = k.shape
    if d != point_pattern.dimension:
        raise ValueError(
            f"k must be of size (n, d) where d=point_pattern.dimension. Given k {k.shape} and d d = {point_pattern.dimension}"
        )
    estimator = select_tapered_estimator(debiased, direct)
    taper = BartlettTaper()
    estimated_sf_k = estimator(k, point_pattern, taper=taper)
    return estimated_sf_k


def allowed_k_scattering_intensity(d, L, k_max=5, meshgrid_shape=None):
    r"""Return a subset of the d-dimensional allowed wavevectors corresponding to a cubic window of length ``L``.

    Args:
        d (int): Dimension of the space containing the point process.

        L (numpy.ndarray): 1d array of size ``d``, where each element correspond to the length of a side of the BoxWindow containing the point process realization.

        k_max (float, optional): Supremum of the components of the allowed wavevectors on which the scattering intensity to be evaluated; i.e., for any allowed wavevector :math:`\mathbf{k}=(k_1,...,k_d)`, :math:`k_i \leq k\_max` for all i. This implies that the maximum of the output vector ``k_norm`` will be approximately equal to the norm of the vector :math:`(k\_max, ... k\_max)`. Defaults to 5.

        meshgrid_shape (tuple, optional): Tuple of length `d`, where each element specifies the number of components over an axis. These axes are crossed to form a subset of :math:`\mathbb{Z}^d` used to construct a set of allowed wavevectors. i.g., if d=2, setting meshgid_shape=(2,3) will construct a meshgrid of allowed wavevectors formed by a vector of 2 values over the x-axis and a vector of 3 values over the y-axis. Defaults to None, which will run the calculation over **all** the allowed wavevectors. Defaults to None.

    Returns:
        tuple (numpy.ndarray, list):
            - k : np.array with ``d`` columns where each row is an allowed wavevector.

    .. proof:definition::

        The set of the allowed wavevectors :math:`\{\mathbf{k}_i\}_i` is defined by

        .. math::

            \{\mathbf{k}_i\}_i = \{\frac{2 \pi}{L} \mathbf{n} ~ ; ~ \mathbf{n} \in (\mathbb{Z}^d)^\ast \}.

        Note that the maximum ``n`` and the number of output allowed wavevectors returned by :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity`, are specified by the input parameters ``k_max`` and ``meshgrid_shape``.

    .. seealso::

        - :py:func:`~structure_factor.tapered_estimators.scattering_intensity`
    """
    assert isinstance(k_max, (float, int))

    n_max = np.floor(k_max * L / (2 * np.pi))  # maximum of ``n``

    #! todo refactoring needed, too complex and duplicated code
    # warnings
    if meshgrid_shape is None:
        warnings.warn(
            message="The computation on all allowed wavevectors may be time-consuming."
        )
    elif (np.array(meshgrid_shape) > (2 * n_max)).any():
        warnings.warn(
            message="Each component of the argument 'meshgrid_shape' should be less than or equal to the cardinality of the (total) set of allowed wavevectors."
        )

    # meshgrid_shape = np.fmin(meshgrid_shape, 2 * n_max)
    # case d=1
    if d == 1:
        if meshgrid_shape is None or (meshgrid_shape > (2 * n_max)):
            n = np.arange(-n_max, n_max + 1, step=1)
            n = n[n != 0]
        else:
            n = np.linspace(-n_max, n_max, num=meshgrid_shape, dtype=int, endpoint=True)
            if np.count_nonzero(n == 0) != 0:
                n = np.linspace(
                    -n_max, n_max, num=meshgrid_shape + 1, dtype=int, endpoint=True
                )
        k = 2 * np.pi * n / L
        k = k.reshape(-1, 1)
    # case d>1
    else:
        if meshgrid_shape is None or (np.array(meshgrid_shape) > (2 * n_max)).any():
            ranges = []
            for n in n_max:
                n_i = np.arange(-n, n + 1, step=1)
                n_i = n_i[n_i != 0]
                ranges.append(n_i)
            n = meshgrid_to_column_matrix(np.meshgrid(*ranges, copy=False))

        else:
            ranges = []
            i = 0
            for s in meshgrid_shape:
                n_i = np.linspace(-n_max[i], n_max[i], num=s, dtype=int, endpoint=True)
                if np.count_nonzero(n_i == 0) != 0:
                    n_i = np.linspace(
                        -n_max[i], n_max[i], num=s + 1, dtype=int, endpoint=True
                    )
                i += 1
                n_i = n_i[n_i != 0]
                ranges.append(n_i)
            n = meshgrid_to_column_matrix(np.meshgrid(*ranges, copy=False))

        k = 2 * np.pi * n / L.T
    return k


def select_tapered_estimator(debiased, direct):
    """Select the tapered estimator of the structure factor.

    Args:
        debiased (bool): Trigger the use of a debiased tapered estimator.
        direct (bool): If ``debiased`` is True, trigger the use of the direct/undirect debiased tapered estimator.

    Returns:
        callable: According to ``debiased`` and ``direct`` return

            - :py:func:`~structure_factor.tapered_estimators.tapered_estimator_debiased_direct`
            - :py:func:`~structure_factor.tapered_estimators.tapered_estimator_debiased_undirect`
            - :py:func:`~structure_factor.tapered_estimators.tapered_estimator_core`
    """
    if debiased:
        if direct:
            return tapered_estimator_debiased_direct
        return tapered_estimator_debiased_undirect
    return tapered_estimator_core


def tapered_estimator_core(k, point_pattern, taper):
    r"""Compute the tapered estimator :math:`S_{TP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

    Args:
        k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the estimator is evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying stationary point process.

        taper (object): Class with static method or instance with method ``.taper(x, window)`` corresponding to :math:`t(x, W)`.

    Returns:
        numpy.ndarray: Evaluation of the tapered estimator :math:`S_{TP}(t, k)` at ``k``.

    .. proof:definition::

        The tapered estimator :math:`S_{TP}(t, k)`, constructed from a realization :math:`\{x_1, \dots, x_N\} \subset \mathbb{R}^d`, is defined by,

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


def tapered_estimator_debiased_direct(k, point_pattern, taper):
    r"""Compute the direct debiased tapered estimator :math:`S_{DDTP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

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


def tapered_estimator_debiased_undirect(k, point_pattern, taper):
    r"""Compute the undirect debiased tapered estimator :math:`S_{UDTP}(t, k)` of the structure factor :math:`S(k)` evaluated at ``k`` w.r.t. the ``taper`` :math:`t` and the realization ``point_pattern`` of the underlying stationary point process.

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

    estimated_sf_k = tapered_estimator_core(k, point_pattern, taper)
    Hk_2 = np.abs(taper.ft_taper(k, window))
    np.square(Hk_2, out=Hk_2)
    estimated_sf_k -= rho * Hk_2
    return estimated_sf_k


def tapered_dft(k, point_pattern, taper):
    r"""Compute the tapered discrete Fourier transform (tapered DFT) associated with ``point_pattern`` evaluated at ``k``, using ``taper`` :math:`t`.

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
        dft (numpy.ndarray): Discrete Fourier transform computed via :py:func:`~structure_factor.tapered_estimators.tapered_dft`.

    Returns:
        numpy.ndarray: ``abs(dft)**2``.
    """
    periodogram = np.zeros_like(dft, dtype=float)
    np.abs(dft, out=periodogram)
    np.square(periodogram, out=periodogram)
    return periodogram
