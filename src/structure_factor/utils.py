"""Collection of secondary functions used in the principal modules."""

import warnings

import numpy as np
from scipy import interpolate, stats
from scipy.special import j0, j1, jn_zeros, jv, y0, y1, yv

# utils for point_processes.py


def get_random_number_generator(seed=None):
    """Turn seed into a np.random.Generator instance."""
    return np.random.default_rng(seed)


# utils for hyperuniformity.py


def _sort_vectors(k, x_k, y_k=None):
    """Sort ``k`` by increasing order and rearranging the associated vectors to ``k``, ``x_k``and ``y_k``.

    Args:
        k (numpy.ndarray): Vector to be sorted by increasing order.
        x_k (numpy.ndarray): Vector of evaluations associated with ``k``.
        y_k (numpy.ndarray, optional): Vector of evaluations associated with ``k``. Defaults to None.

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray): ``k`` sorted by increasing order and the associated vectors ``x_k``and ``y_k``.
    """
    sort_index_k = np.argsort(k)
    k_sorted = k[sort_index_k]
    x_k_sorted = x_k[sort_index_k]
    if y_k is not None:
        y_k_sorted = y_k[sort_index_k]
        return k_sorted, x_k_sorted, y_k_sorted
    return k_sorted, x_k_sorted, y_k


# utils for hyperuniformity.py and structure_factor.py


def _bin_statistics(x, y, **params):
    """Divide ``x`` into bins and evaluate the mean and the standard deviation of the corresponding elements of ``y`` over each bin.

    Args:
        x (numpy.ndarray): Vector of data.
        y (numpy.ndarray): Vector of data associated with the vector ``x``.

    Keyword Args:
        params (dict): Keyword arguments (except ``"x"``, ``"values"`` and ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray):

            - ``bin_centers``: Vector of centers of the bins associated to ``x``.
            - ``bin_mean``: Vector of means of ``y`` over the bins.
            - ``std_mean``: Vector of standard deviations of ``y`` over the bins.
    """
    bin_mean, bin_edges, _ = stats.binned_statistic(x, y, statistic="mean", **params)
    bin_centers = np.convolve(bin_edges, np.ones(2), "valid")
    bin_centers /= 2
    count, _, _ = stats.binned_statistic(x, y, statistic="count", **params)
    bin_std, _, _ = stats.binned_statistic(x, y, statistic="std", **params)
    bin_std /= np.sqrt(count)
    return bin_centers, bin_mean, bin_std


# utils for tranform.py


def bessel1(order, x):
    """Evaluate `Bessel function of the first kind <https://en.wikipedia.org/wiki/Bessel_function>`_."""
    if order == 0:
        return j0(x)
    if order == 1:
        return j1(x)
    return jv(order, x)


def bessel1_zeros(order, nb_zeros):
    """Evaluate zeros of the `Bessel function of the first kind <https://en.wikipedia.org/wiki/Bessel_function>`_."""
    return jn_zeros(order, nb_zeros)


def bessel2(order, x):
    """Evaluate `Bessel function of the second kind <https://en.wikipedia.org/wiki/Bessel_function>`_."""
    if order == 0:
        return y0(x)
    if order == 1:
        return y1(x)
    return yv(order, x)


# utils for pair_correlation_function.py


def set_nan_inf_to_zero(array, nan=0, posinf=0, neginf=0):
    """Set nan, posinf, and neginf values of ``array`` to the corresponding input arguments. Defaults to zero."""
    return np.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf)


def _extrapolate_pcf(x, r, pcf_r, **params):
    """Interpolate pcf_r for x=<r_max and set to 1 for x>r_max.

    Args:
        x (numpy.ndarray): Points on which the pair correlation function is to be evaluated.

        r (numpy.ndarray): Vector of the radius on with the pair correlation function was evaluated.

        pcf_r (numpy.ndarray): Vector of evaluations of the pair correlation function corresponding to ``r``.

    Returns:
        numpy.ndarray: evaluation of the extrapolated pair correlation function on ``x``.
    """
    r_max = np.max(r)  # maximum radius
    pcf = np.zeros_like(x)
    params.setdefault("fill_value", "extrapolate")

    mask = x > r_max
    if np.any(mask):
        pcf[mask] = 1.0
        np.logical_not(mask, out=mask)
        pcf[mask] = interpolate.interp1d(r, pcf_r, **params)(x[mask])
    else:
        pcf = interpolate.interp1d(r, pcf_r, **params)(x)

    return pcf


# utils for structure_factor.py


def norm(k):
    return np.linalg.norm(k, axis=-1)


def meshgrid_to_column_matrix(X):
    r"""Transform output ``X`` of numpy.meshgrid to a 2d numpy array with columns formed by flattened versions of the elements of ``X``.

    .. code-block:: python

        np.column_stack([x.ravel() for x in X])

    Args:
        X (list): output of numpy.meshgrid.

    Returns:
        np.ndarray: 2d array.
    """
    return np.column_stack([x.ravel() for x in X])


def allowed_wave_vectors(d, L, k_max=5, meshgrid_shape=None):
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

        Note that the maximum ``n`` and the number of output allowed wavevectors returned by :py:meth:`allowed_wave_vectors`, are specified by the input parameters ``k_max`` and ``meshgrid_shape``.
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
