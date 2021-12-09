import numpy as np
import structure_factor.utils as utils


def tapered_periodogram(k, points, taper, intensity):
    r"""Compute the spectral estimator :math:`S_h(k)` associated to the taper :math:`h`.


    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable, or float): taper :math:`h`.

        intensity (float): intensity of the point process :math:`\rho`.

    Returns:
        numpy.ndarray: Evaluation(s) of the spectral estimator :math:`S_h(k)` at ``k``.

    .. proof:definition::

        The spectral estimator :math:`\widehat{S}_{h}`, of a realization of points :math:`\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathbb{R}^d`, is defined by,

        .. math::

            \widehat{S}_{h}(\mathbf{k}) =
                    \frac{1}{\rho} \left\lvert
                    \sum_{j=1}^N
                        h(x_j)
                        \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector.
    """
    dft = tapered_DFT(k, points, taper)
    # periodogram = (1/rho) * | dft |^2
    periodogram = np.zeros_like(dft, dtype=float)
    np.abs(dft, out=periodogram)
    np.square(periodogram, out=periodogram)
    periodogram /= intensity
    return periodogram


def tapered_DFT(k, points, taper):
    r"""Compute the discrete Fourier transform associated to points (x) of the taper h on k i.e.,  sum_x h(x) exp(- i <k, x>).

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable or float): taper :math:`h`.

    Returns:
        numpy.ndarray: Evaluation(s) of the DFT of the taper h
    """
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
    hx = taper(X) if callable(taper) else taper
    hx_exp_ikx *= hx

    dft = np.sum(hx_exp_ikx, axis=1)
    return dft


def scattering_intensity(k, points, window_volume, intensity):
    r"""Compute the scattering intensity of ``points`` at each wavevector in ``k``. Particular case of :py:meth:`tapered_periodogram` for the constant taper 1/|W|, where |W| is the volume of the window containing ``points``.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        window_volume (float): Volume of the window containing the realization of the point process.

        intensity (float): Intensity of the point process :math:`\rho`.

    Returns:
        numpy.ndarray: Evaluation(s) of the scattering intensity on ``k``.

    .. proof:definition::

        The scattering intensity :math:`\widehat{S}_{SI}`, of a realization of points :math:`\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathbb{R}^d`, is defined by,

        .. math::

            \widehat{S}_{SI}(\mathbf{k}) =
                \frac{1}{N}\left\lvert
                    \sum_{j=1}^N
                        \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector.
        Equivalently, to prevent additional bias the factor N, in  :math:`\frac{1}{N}` could be replaced by the product of, the intensity of the point process and the volume of the window containing the realization :math:`\{\mathbf{x}_i\}_{i=1}^N`.
    """
    h0 = 1.0 / np.sqrt(window_volume)
    return tapered_periodogram(k, points, h0, intensity)


def debiased_tapered_periodogram(k, points, taper, intensity, ft_taper):
    r"""Debiased periodogram of a point process (x) for a specific taper (h) i.e., computes abs(sum_x h(x) exp(- i <k, x>) - rho*F(h)(k))**2.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable or float): taper :math:`h`.

        intensity (float): Intensity of the point process :math:`\rho`.

        ft_taper (np.ndarray): Fourier transform of `taper` evaluated on `k`

    Returns:
        numpy.ndarray: Evaluation(s) of the debiased periodogram on ``k``.
    """

    # dft of the taper
    dft = tapered_DFT(k, points, taper)
    debiased_periodogram = np.zeros_like(dft, dtype=float)
    # removing biase
    debiased_periodogram = dft - intensity * ft_taper

    # debiased periodogram
    np.abs(debiased_periodogram, out=debiased_periodogram)
    np.square(debiased_periodogram, out=debiased_periodogram)
    debiased_periodogram /= intensity
    return np.real(debiased_periodogram)


def debiased_scattering_intensity(k, points, intensity, window):
    r"""Debiased scattering intensity. Particular case of :py:meth:`debiased_tapered_periodogram` for the constant taper 1/|W|, where |W| is the volume of the window containing ``points``.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        intensity (float): Intensity of the point process :math:`\rho`.

        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Support window of the realization of the point process.

    Returns:
        numpy.ndarray: Evaluation(s) of the debiased scattering intensity on ``k``.
    """
    # taper
    h0 = 1.0 / np.sqrt(window.volume)
    # Fourier transform of the taper
    ft_h0 = utils.ft_h0(k, window)
    return np.real(debiased_tapered_periodogram(k, points, h0, intensity, ft_h0))
