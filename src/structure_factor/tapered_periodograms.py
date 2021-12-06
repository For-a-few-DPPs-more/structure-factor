import numpy as np


# ! Draft of code
# todo play with and test it
def tapered_periodogram(k, points, taper=None):
    r"""Compute the spectral estimator :math:`S_h(k)` associated to the taper :math:`h`.

    .. proof:definition::

        The spectral estimator :math:`\widehat{S}_{h}`, of a realization of points :math:`\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathbb{R}^d`, is defined by,

        .. math::

            \widehat{S}_{h}(\mathbf{k}) =
                \left\lvert
                    \sum_{j=1}^N
                        h(x_j)
                        \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector.

    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

        taper (callable, optional): taper :math:`h`. Defaults to None.
        When ``taper`` is None, it is set to :math:`h(x) = \frac{1}{\sqrt{N}}` which corresponds to the scattering intensity or Bartlett's periodogram estimator.

    Returns:
        numpy.ndarray: Evaluation(s) of the spectral estimator :math:`S_h(k)` at ``k``.
    """
    K = np.atleast_2d(k)
    X = np.atleast_2d(points)
    nb_k, dim_k = K.shape
    nb_x, dim_x = X.shape

    # Compute | sum_x h(x) exp(- i <k, x>) |^2
    hx_exp_ikx = np.zeros((nb_k, nb_x), dtype=complex)
    # i <k, x>
    np.dot(K, X.T, out=hx_exp_ikx.imag)
    # - i <k, x>
    np.conj(hx_exp_ikx, out=hx_exp_ikx)
    # exp(- i <k, x>)
    np.exp(hx_exp_ikx, out=hx_exp_ikx)

    if taper is not None:
        # h(x) exp(- i <k, x>)
        hx = taper(X)
        hx_exp_ikx *= hx

    periodogram = np.zeros(nb_k, dtype=float)
    # | sum_x h(x) exp(- i <k, x>) |
    np.abs(np.sum(hx_exp_ikx, axis=1), out=periodogram)
    # | sum_x h(x) exp(- i <k, x>) |^2
    np.square(periodogram, out=periodogram)

    if taper is None:  # h(x) = 1 / sqrt(nb_x)
        periodogram /= nb_x

    return periodogram


def bartlett_periodogram(k, points):
    # taper = lambda(x): 1.0 / np.sqrt(points.shape[0])
    return tapered_periodogram(k, points, taper=None)


def scattering_intensity(k, points):
    return bartlett_periodogram(k, points)
