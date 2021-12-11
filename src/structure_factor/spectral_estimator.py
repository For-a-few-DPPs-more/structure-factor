import numpy as np

# ? I find this not very explicit to call functions periodograms in a file called estimators where the returned values are not periodograms but rescaled periodograms that correspond to estimators of the structure factor...
# ? I'd rather like to rename the file to periodograms, where the different functions indeed return periodograms to be rescaled in the corresponding StructureFactor method


def tapered_dft(k, point_pattern, taper):
    r"""Compute the discrete Fourier transform associated to points (x) of the taper h on k i.e.,  sum_x h(x) exp(- i <k, x>).

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable, np.array, float): taper :math:`h(X, W)` function of 2 variables: the realization of the point process `X` and its support window `W`, or the array containing the evaluations of the taper on the points of the realization (which may be reduced to a float if the taper is constant).

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
    hx = taper(X, window) if callable(taper) else taper
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

        taper (callable, np.array, float): taper :math:`h(X, W)` function of 2 variables: the realization of the point process `X` and its support window `W`, or the array containing the evaluations of the taper on the points of the realization (which may be reduced to a float if the taper is constant).

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
    intensity = point_pattern.intensity
    dft = tapered_dft(k, point_pattern, taper)
    estimator = periodogram_from_dft(dft)
    estimator /= intensity
    return estimator


#! add test
def debiased_tapered_periodogram(k, point_pattern, taper, ft_taper):
    r"""Debiased periodogram of a point process (x) for a specific taper (h) i.e., computes abs(sum_x h(x) exp(- i <k, x>) - rho*F(h)(k))**2.

    Args:
        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable, np.array, float): taper :math:`h(X, W)` function of 2 variables: the realization of the point process `X` and its support window `W`, or the array containing the evaluations of the taper on the points of the realization (which may be reduced to a float if the taper is constant)

        ft_taper (callable, np.array, float): Fourier transform of `taper`.  :math:`F(h)(k, W)` function of 2 variables: the wavevector(s) `k` and the support window of the realization of the point process `W`, or an array containing the evaluations of the Fourier transform of the taper on `k` (which may be reduced to a float if the taper is constant)

    Returns:
        numpy.ndarray: Evaluation(s) of the debiased periodogram on ``k``.
    """
    intensity = point_pattern.intensity
    window = point_pattern.window

    # Debiased dft
    dft = tapered_dft(k, point_pattern, taper)
    dft -= intensity * ft_taper(k, window)

    estimator = periodogram_from_dft(dft)
    estimator /= intensity
    return estimator


#! add test
def undirect_debiased_tapered_periodogram(k, point_pattern, taper, ft_taper):
    window = point_pattern.window
    intensity = point_pattern.intensity
    # tapered periodogram
    periodogram = tapered_periodogram(k, point_pattern, taper)
    Hk_2 = np.abs(ft_taper(k, window))
    np.square(Hk_2, out=Hk_2)
    periodogram -= intensity * Hk_2
    return periodogram
