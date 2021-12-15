import numpy as np
from scipy.spatial.distance import pdist

from structure_factor.spatial_windows import UnitBallWindow
from structure_factor.utils import bessel1

# ? I find this not very explicit to call functions periodograms in a file called estimators where the returned values are not periodograms but rescaled periodograms that correspond to estimators of the structure factor...
# ? I'd rather like to rename the file to periodograms, where the different functions indeed return periodograms to be rescaled in the corresponding StructureFactor method


def tapered_dft(k, point_pattern, taper):
    r"""Compute the tapered discrete Fourier transform associated  ``point_pattern`` evaluated at wavevectors ``k``, using ``taper``
    .. math::

        \sum_{j=1}^N h(x_j) exp(- i <k, x_j>).

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (AbstractTaper): class with method ``taper`` :math:`h(X, W)`.

    Returns:
        numpy.ndarray: Evaluation(s) of the DFT of the taper h
    """
    points = point_pattern.points
    window = point_pattern.window
    K = np.atleast_2d(k)
    X = np.atleast_2d(points)
    nb_k, _ = K.shape
    nb_x, _ = X.shape

    # dft = sum_x h(x) exp(- i <k, x>)
    hx_exp_ikx = np.zeros((nb_k, nb_x), dtype=complex)
    # i <k, x>
    hx_exp_ikx.imag = np.dot(K, X.T)
    # - i <k, x>
    np.conj(hx_exp_ikx, out=hx_exp_ikx)
    # exp(- i <k, x>)
    np.exp(hx_exp_ikx, out=hx_exp_ikx)
    # h(x) exp(- i <k, x>)
    hx = taper.taper(X, window)
    hx_exp_ikx *= hx

    dft = np.sum(hx_exp_ikx, axis=1)
    return dft


def periodogram_from_dft(dft):
    periodogram = np.zeros_like(dft, dtype=float)
    np.abs(dft, out=periodogram)
    np.square(periodogram, out=periodogram)
    return periodogram


def tapered_periodogram(k, point_pattern, taper):
    r"""Compute the spectral estimator :math:`S_h(k)` associated to the taper :math:`h`.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type PointPattern containing a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.

        taper (AbstractTaper): class with method ``.taper(x, W)`` :math:`h(x, W)`, where :math:`W` corresponds to ``point_pattern.window``.

    Returns:
        numpy.ndarray: Evaluation(s) of the spectral estimator :math:`S_h(k)` at ``k``.

    .. proof:definition::

        The spectral estimator :math:`\widehat{S}_{h}`, of a realization of points :math:`\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathbb{R}^d`, is defined by,

        .. math::

            \widehat{S}_{h}(\mathbf{k}) =
            \frac{1}{\rho}
            \left\lvert
                \sum_{j=1}^N
                h(x_j, W)
                \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
            \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector.
    """
    rho = point_pattern.intensity
    dft = tapered_dft(k, point_pattern, taper)
    estimator = periodogram_from_dft(dft)
    estimator /= rho
    return estimator


#! add test
def debiased_tapered_periodogram(k, point_pattern, taper):
    r"""Debiased tapered periodogram computed from ``point_pattern`` evaluated at wavevectors ``k``, using ``taper``

    .. math::

        \widehat{S}_{h}(\mathbf{k}) =
            \frac{1}{\rho} \left\lvert
            \sum_{j=1}^N
                h(x_j, W)
                \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                - \rho * F[h(\cdot, W)](k)
            \right\rvert^2

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (AbstractTaper): class with two methods:

            - ``.taper(x, window)`` :math:`h(x, W)`,
            - ``.ft_taper(k, window)``, Fourier transform :math:`\mathcal{F}[h(\dot, W)](k)` of the taper.  :math:`F(h)(k, W)`,

            where :math:`W` corresponds to ``point_pattern.window``.
    Returns:
        numpy.ndarray: Evaluation(s) of the debiased periodogram on ``k``.
    """
    rho = point_pattern.intensity
    window = point_pattern.window

    # Debiased dft
    dft = tapered_dft(k, point_pattern, taper)
    dft -= rho * taper.ft_taper(k, window)

    estimator = periodogram_from_dft(dft)
    estimator /= rho
    return estimator


#! add test
def undirect_debiased_tapered_periodogram(k, point_pattern, taper):
    window = point_pattern.window
    rho = point_pattern.intensity

    periodogram = tapered_periodogram(k, point_pattern, taper)
    Hk_2 = np.abs(taper.ft_taper(k, window))
    np.square(Hk_2, out=Hk_2)
    periodogram -= rho * Hk_2
    return periodogram


def multitapered_periodogram(k, point_pattern, *tapers, debiased=False, undirect=False):
    if debiased:
        if undirect:
            periodogram = undirect_debiased_tapered_periodogram
        periodogram = debiased_tapered_periodogram
    else:
        periodogram = tapered_periodogram

    multi_periodogram = np.zeros(k.shape[0], dtype=float)
    for taper in tapers:
        multi_periodogram += periodogram(k, point_pattern, taper)
    multi_periodogram /= len(tapers)
    return multi_periodogram


def isotropic_estimator(k, point_pattern):
    # ! the current implem may take some time when there are a lot of wave vectors and points
    window = point_pattern.window
    d = window.dimension
    unit_ball = UnitBallWindow(np.zeros(d))

    X = np.atleast_2d(point_pattern.points)
    norm_x_y = pdist(X, metric="euclidean")
    K = np.atleast_2d(k)
    norm_k = np.linalg.norm(K, axis=1)

    k_xy = np.multiply.outer(norm_k, norm_x_y)
    order = d / 2 - 1
    J_k_xy = bessel1(order, k_xy)

    estimator = np.zeros_like(norm_k)
    # for i, k_ in enumerate(norm_k):
    #     estimator[i] = bessel1(order, k_ * norm_x_y).sum()
    if order > 0:
        np.power(k_xy, order, out=k_xy)
        J_k_xy /= k_xy
    np.sum(J_k_xy, axis=1, out=estimator)

    surface, volume = unit_ball.surface, window.volume
    rho = point_pattern.intensity
    estimator *= (2 * np.pi) ** (d / 2) / (surface * volume * rho)
    estimator += 1

    return norm_k, estimator
