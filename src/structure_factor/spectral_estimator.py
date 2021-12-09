import numpy as np


# ! Draft of code
# todo play with and test it
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
    K = np.atleast_2d(k)
    nb_k, _ = K.shape
    # Compute sum_x h(x) exp(- i <k, x>)
    hx_exp_ikx = tapered_DFT(k, points, taper)

    periodogram = np.zeros(nb_k, dtype=float)
    # | sum_x h(x) exp(- i <k, x>) |
    np.abs(hx_exp_ikx, out=periodogram)
    # (1/rho)*| sum_x h(x) exp(- i <k, x>) |^2
    np.square(periodogram, out=periodogram)

    return periodogram / intensity


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

    hx_exp_ikx = np.zeros((nb_k, nb_x), dtype=complex)
    # i <k, x>
    # np.dot(K, X.T, out=hx_exp_ikx.imag)
    hx_exp_ikx.imag = np.dot(K, X.T)
    # - i <k, x>
    np.conj(hx_exp_ikx, out=hx_exp_ikx)
    # exp(- i <k, x>)
    np.exp(hx_exp_ikx, out=hx_exp_ikx)

    # taper
    if callable(taper):
        hx = taper(X)
    else:
        hx = taper
    hx_exp_ikx *= hx
    # periodogram
    periodogram = np.zeros(nb_k, dtype=float)
    #  sum_x h(x) exp(- i <k, x>)
    periodogram = np.sum(hx_exp_ikx, axis=1)
    return periodogram


def scattering_intensity(k, points, window_volume, intensity):
    r"""Compute the scattering intensity of ``points`` at each wavevector in ``k``.

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
    taper = 1 / np.sqrt(window_volume)
    return tapered_periodogram(k, points, taper, intensity)
