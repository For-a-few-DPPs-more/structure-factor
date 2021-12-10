import numpy as np

import structure_factor.utils as utils


def undirected_debiased_tapered_periodogram(k, point_pattern, taper, ft_taper):
    rho = point_pattern.intensity
    window = point_pattern.window
    perio = tapered_periodogram(k, point_pattern, taper=taper)
    perio -= rho ** 2 * ft_taper(k, window)
    return perio / rho


def debiased_tapered_periodogram(k, point_pattern, taper, ft_taper):
    rho = point_pattern.intensity
    window = point_pattern.window
    dft = tapered_DFT(k, point_pattern, taper=taper)
    dft -= rho * ft_taper(k, window)
    periodogram = periodogram_from_dft(dft)
    return periodogram / rho


def tapered_periodogram(k, point_pattern, taper):
    rho = point_pattern.intensity
    dft = tapered_DFT(k, point_pattern, taper=taper)
    periodogram = periodogram_from_dft(dft)
    return periodogram / rho


def periodogram(k, point_pattern, taper=None, ft_taper=None):
    r"""Compute the spectral estimator :math:`S_h(k)` associated to the taper :math:`h`.

    [extended_summary]

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        point_pattern ([type]): [description]

        taper (callable | float, optional): taper :math:`h`. Defaults to None.
        If None, fall back on scattering intensity estimator, i.e., :math:`h(x) = \frac{1}{\sqrt{|W|}}`.

        ft_taper ([type], optional): Fourier transform of ``taper`` :math:`H(k)=\mathcal{F}[h](k)`. Defaults to None. If not None, the debiased version of the periodogram is computed, otherwise the term :math:`- \rho^2 H(k)` is set to 0.

    Returns:
        numpy.ndarray: Evaluation(s) of the spectral estimator :math:`S_h(k)` at ``k``.

    .. proof:definition::

        The spectral estimator :math:`\widehat{S}_{h}`, of a realization of points :math:`\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathbb{R}^d`, is defined by,

        .. math::

            \widehat{S}_{h}(\mathbf{k}) =
                    \frac{1}{\rho}
                    \left\lvert
                    \sum_{x\in \mathcal{X}}
                        h(x)
                        \exp(- i\langle \mathbf{k}, \mathbf{x} \rangle)
                        - \rho^2 H(k)
                    \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector.
    """

    r"""Compute the spectral estimator :math:`S_h(k)` associated to the taper :math:`h`.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        point_pattern

        taper (callable, or float): taper :math:`h`.

        ft_taper (callable, or float):

    Returns:
        numpy.ndarray: Evaluation(s) of the spectral estimator :math:`S_h(k)` at ``k``.
    """
    rho = point_pattern.intensity
    window = point_pattern.window
    dft = tapered_DFT(k, point_pattern, taper=taper)
    if taper is not None and ft_taper is not None:  # debias the dft
        dft -= rho * ft_taper(k, window)
    periodogram = periodogram_from_dft(dft)
    return periodogram / rho


def periodogram_from_dft(dft):
    # periodogram = |dft|^2
    periodogram = np.zeros_like(dft, dtype=float)
    np.abs(dft, out=periodogram)
    np.square(dft, out=periodogram)
    return periodogram


def tapered_DFT(k, point_pattern, taper=None):
    r"""Compute the discrete Fourier transform associated to point_pattern (x) of the taper h on k i.e.,  sum_x h(x) exp(- i <k, x>).

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``point_pattern``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        point_pattern (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable or float): taper :math:`h`.

    Returns:
        numpy.ndarray: Evaluation(s) of the DFT of the taper h
    """
    K = np.atleast_2d(k)
    X = np.atleast_2d(point_pattern.points)
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
    if taper is None:
        hx = 1.0 / np.sqrt(point_pattern.window.volume)
    elif callable(taper):
        hx = taper(X)
    else:
        hx = taper
    hx_exp_ikx *= hx

    dft = np.sum(hx_exp_ikx, axis=1)
    return dft


def scattering_intensity(k, point_pattern, debiased=False):
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
    window = point_pattern.window
    taper = 1.0 / np.sqrt(window.volume)
    ft_taper = utils.ft_h0(k, window) if debiased else None
    return periodogram(k, point_pattern, taper, ft_taper)
