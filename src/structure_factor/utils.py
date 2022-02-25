"""Collection of secondary functions used in the principal modules."""

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
