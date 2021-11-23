#!/usr/bin/env python3
# coding=utf-8

# import pandas as pd
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, stats
from scipy.special import j0, j1, jn_zeros, jv, y0, y1, yv


# utils for the class Hyperuniformity
def _sort_vectors(k, x_k, y_k=None):
    """Sort ``k`` by increasing order and reorder the associated vectors to ``k``, ``x_k``and ``y_k``.

    Args:
        k (np.array): Vector to be sorted by increasing order.
        x_k (np.array): Vector of evaluations associated to ``k``.
        y_k (np.array, optional): Defaults to None. Vector of evaluations associated to ``k``.

    Returns:
        (np.array, np.array, np.array): ``k`` sorted by increasing order and the associated ``x_k``and ``y_k``.
    """
    sort_index_k = np.argsort(k)
    k_sorted = k[sort_index_k]
    x_k_sorted = x_k[sort_index_k]
    if y_k is not None:
        y_k_sorted = y_k[sort_index_k]
        return k_sorted, x_k_sorted, y_k_sorted
    return k_sorted, x_k_sorted, y_k


def set_nan_inf_to_zero(array, nan=0, posinf=0, neginf=0):
    """Set nan, posinf and neginf values of ``array`` to 0."""
    return np.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf)


def get_random_number_generator(seed=None):
    """Turn seed into a np.random.Generator instance."""
    return np.random.default_rng(seed)


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


####### Theoretical structure factors


def pair_correlation_function_ginibre(x):
    return 1.0 - np.exp(-(x ** 2))


def structure_factor_poisson(k):
    return np.ones_like(k)


def structure_factor_ginibre(k):
    return 1.0 - np.exp(-(k ** 2) / 4)


def _reshape_meshgrid(X):
    r"""Reshape list of meshgrids as np.ndarray of columns, where each column is associated to an element (meshgrid) of the list.

    Args:
        X (list): List of meshgrids.

    Returns:
        np.ndarray: np.ndarray where each meshgrid of the original list is stacked as a column.
    """
    T = []
    d = len(X)
    for i in range(0, d):
        T.append(X[i].ravel())
    n = np.column_stack(T)  # stack in d columns
    return n


###### utils for the class StructureFactor


def allowed_wave_vectors(d, L, k_max, meshgrid_shape=None):
    r"""Return a subset of the d-dimentional allowed wave vectors corresponding to a cubic window of length ``L``.

    Args:

        d (int): Dimension of the space containing the point process.

        L (float): Length of the cubic window containing the point process realization.

        k_max (float): Maximum component of the waves vectors i.e., for any allowed wave vector :math:`\mathbf{k}=(k_1,...,k_d)`, :math:`k_i \leq k\_max` for all i. This implies that the maximum wave vector will be :math:`(k\_max, ... k\_max)`.

        meshgrid_shape (tuple, optional): Tuple of length `d`, where each element specify the number of components over the corresponding axis. It consists of the associated size of the meshgrid of allowed waves. For example for :math:`d=2`, letting meshgid_shape=(2,3) gives a meshgrid of allowed waves formed by a vector of 2 values over the x-axis and a vectors of 3 values over the y-axis. Defaults to None.

    Returns:
        tuple (np.ndarray, list):
            - k : np.array with ``d`` columns where each row is an allowed wave vector.
            - K : list of meshgrids, (the elements of the list correspond to the 2D respresentation of the components of the wave vectors, i.e., a 2D representation of the vectors of allowed values ``k``). For example in dimension 2, if K =[X,Y] then X is the 2D representation of the x coordinates of the allowed wave vectors ``k`` i.e., the representation as meshgrid.

    .. proof:definition::

        The set of the allowed wavevectors :math:`\{\mathbf{k}_i\}_i` is defined by

        .. math::

            \{\mathbf{k}_i\}_i = \{\frac{2 \pi}{L} \mathbf{n} ~ ; ~ \mathbf{n} \in (\mathbb{Z}^d)^\ast \}.

        For plotting purposes, we typically use a subset of allowed wavevectors. The maximum norm and the number of output allowed wavevectors returned by :py:meth:`allowed_wave_vectors`, are specified by the input parameters ``k_max`` and ``meshgrid_shape``.
    """
    K = None
    n_max = np.floor(k_max * L / (2 * np.pi))  # maximum of ``n``

    # warnings
    if meshgrid_shape is None:
        warnings.warn(message="Taking all allowed wave vectors may be time consuming.")
    elif (np.array(meshgrid_shape) > (2 * n_max)).any():
        warnings.warn(
            message="meshgrid_shape should be smaller than that of the complete meshgrid of allowed wavevectors."
        )

    if meshgrid_shape is None or (np.array(meshgrid_shape) > (2 * n_max)).any():
        n_all = ()
        n_i = np.arange(-n_max, n_max + 1, step=1)
        n_i = n_i[n_i != 0]
        n_all = (n_i for i in range(0, d))
        X = np.meshgrid(*n_all, copy=False)
        K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wave vectors
        n = _reshape_meshgrid(X)  # reshape allowed vectors as d columns

    else:
        if d == 1:
            n = np.linspace(-n_max, n_max, num=meshgrid_shape, dtype=int, endpoint=True)
            if np.count_nonzero(n == 0) != 0:
                n = np.linspace(
                    -n_max, n_max, num=meshgrid_shape + 1, dtype=int, endpoint=True
                )

        else:
            n_all = []
            for s in meshgrid_shape:
                n_i = np.linspace(-n_max, n_max, num=s, dtype=int, endpoint=True)
                if np.count_nonzero(n_i == 0) != 0:
                    n_i = np.linspace(
                        -n_max, n_max, num=s + 1, dtype=int, endpoint=True
                    )
                n_i = n_i[n_i != 0]
                n_all.append(n_i)

            X = np.meshgrid(*n_all, copy=False)
            K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wave vectors
            n = _reshape_meshgrid(X)  # reshape allowed vectors as d columns

    k = 2 * np.pi * n / L
    return k, K


def compute_scattering_intensity(k, points):
    r"""Compute the scattering intensity of ``points`` at each wavevector in ``k``.

    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimesion of the space containing ``points``). Each row is a wave vector on which the scattering intensity to be evaluated.

        points (np.ndarray): np.ndarray of d columns where each row is a point from the realization of the point process.

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
    """
    n = points.shape[0]  # number of points
    if points.shape[1] != k.shape[1]:
        raise ValueError("k and points should have same number of columns")

    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, points.T)), axis=1)))
    si /= n

    # reshape the output

    return si


def _bin_statistics(x, y, **params):
    """Divide ``x`` into bins and evaluate the mean and the standard deviation of the corresponding elements of ``y`` over each bin.

    Args:
        x (np.array): Vector of data.
        y (np.array): Vector of data associated to the vector ``x``.

    Keyword args:
        params (dict): Keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray):
            - ``bin_centers``: Vector of centers of the bins associated to ``x``.
            - ``bin_mean``: Vector of means of ``y`` over the bins.
            - ``std_mean``: Vector of standard deviation of ``y`` over the bins.
    """
    bin_mean, bin_edges, _ = stats.binned_statistic(x, y, statistic="mean", **params)
    bin_centers = np.convolve(bin_edges, np.ones(2), "valid")
    bin_centers /= 2
    count, _, _ = stats.binned_statistic(x, y, statistic="count", **params)
    bin_std, _, _ = stats.binned_statistic(x, y, statistic="std", **params)
    bin_std /= np.sqrt(count)
    return bin_centers, bin_mean, bin_std


def plot_poisson(x, axis, c="k", linestyle=(0, (5, 10)), label="Poisson"):
    r"""Plot the pair correlation function :math:`g_{poisson}` and the structure factor :math:`S_{poisson}` corresponding to the Poisson point process.

    Args:
        x (np.array): x coordinate.

        axis (matplotlib.axis): Axis on which to add the plot.

        c (str, optional): Color of the plot. see `matplotlib <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ . Defaults to "k".

        linestyle (tuple, optional): Linstyle of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_. Defaults to (0, (5, 10)).

        label (str, optional): Specification of the label of the plot. Defaults to "Poisson".

    Returns:
        matplotlib.plot: Plot of the pair correlation function and the structure factor of the Poisson point process over ``x``.

    """
    axis.plot(x, np.ones_like(x), c=c, linestyle=linestyle, label=label)
    return axis


def plot_summary(x, y, axis, label=r"mean $\pm$ 3 $\cdot$ std", **binning_params):
    """Loglog plot the summary results of :py:func:`~structure_factor.utils._bin_statistics` i.e., means and errors bars (3 standard deviations)."""
    bin_centers, bin_mean, bin_std = _bin_statistics(x, y, **binning_params)
    axis.loglog(bin_centers, bin_mean, "b.")
    axis.errorbar(
        bin_centers,
        bin_mean,
        yerr=3 * bin_std,  # 3 times the standard deviation
        fmt="b",
        lw=1,
        ecolor="r",
        capsize=3,
        capthick=1,
        label=label,
        zorder=4,
    )

    return axis


def plot_exact(x, y, axis, label):
    """Loglog plot of a callable function ``y`` evaluated on the vector ``x``."""
    axis.loglog(x, y(x), "g", label=label)
    return axis


def plot_approximation(x, y, axis, label, color, linestyle, marker, markersize):
    """Loglog plot of ``y`` w.r.t. ``x``."""
    axis.loglog(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        label=label,
        markersize=markersize,
    )
    return axis


def plot_si_showcase(
    norm_k,
    si,
    axis=None,
    exact_sf=None,
    error_bar=False,
    file_name="",
    **binning_params
):
    """Loglog plot of the results of the scattering intensity :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, with the means and error bars over specific number of bins found via :py:func:`~structure_factor.utils._bin_statistics`."""
    norm_k = norm_k.ravel()
    si = si.ravel()
    if axis is None:
        _, axis = plt.subplots(figsize=(8, 6))

    plot_poisson(norm_k, axis=axis)
    if exact_sf is not None:
        plot_exact(norm_k, exact_sf, axis=axis, label=r"Exact $S(k)$")

    plot_approximation(
        norm_k,
        si,
        axis=axis,
        label=r"$\widehat{S}_{SI}$",
        color="grey",
        linestyle="",
        marker=".",
        markersize=1.5,
    )

    if error_bar:
        plot_summary(norm_k, si, axis=axis, **binning_params)

    axis.set_xlabel(r"Wavenumber ($||\mathbf{k}||$)")
    axis.set_ylabel(r"Structure factor ($S(k)$)")
    axis.legend(loc=4, framealpha=0.2)

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_si_imshow(norm_k, si, axis, file_name):
    """Color level plot, centered on zero."""
    if axis is None:
        _, axis = plt.subplots(figsize=(14, 8))
    if len(norm_k.shape) < 2:
        raise ValueError(
            "the scattering intensity should be evaluated on a meshgrid or choose plot_type = 'plot'. "
        )
    else:
        log_si = np.log10(si)
        m, n = log_si.shape
        m /= 2
        n /= 2
        f_0 = axis.imshow(
            log_si,
            extent=[-n, n, -m, m],
            cmap="PRGn",
        )
        plt.colorbar(f_0, ax=axis)
        # axis.title.set_text("Scattering intensity")

        if file_name:
            fig = axis.get_figure()
            fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_si_all(
    point_pattern,
    norm_k,
    si,
    exact_sf=None,
    error_bar=False,
    file_name="",
    window_res=None,
    **binning_params
):
    """Construct 3 subplots: point pattern, associated scattering intensity plot, associated scattering intensity color level (only for 2D point processes)."""
    figure, axes = plt.subplots(1, 3, figsize=(24, 6))

    point_pattern.plot(axis=axes[0], window_res=window_res)
    plot_si_showcase(
        norm_k,
        si,
        axes[1],
        exact_sf,
        error_bar,
        file_name="",
        **binning_params,
    )
    plot_si_imshow(norm_k, si, axes[2], file_name="")

    if file_name:
        figure.savefig(file_name, bbox_inches="tight")

    return axes


def _compute_k_min(r_max, step_size):
    """Estimate threshold of confidence for the approximation of the Hankel transform using Ogata method. i.e., minimum confidence k for which the approximation of the Hankel transform by Ogata quadrature is doable :cite:`Oga05`.

    Args:
        r_max (float): Maximum radius on which the input function :math:`f` to be Hankel transformed was evaluated before the interpolation.

        step_size (float): Stepsize used in the quadrature of Ogata.

    Returns:
        float: Upper bound of k.
    """
    return (2.7 * np.pi) / (r_max * step_size)


def plot_pcf(pcf_dataframe, exact_pcf, file_name, **kwargs):
    """Plot DataFrame result."""
    axis = pcf_dataframe.plot.line(x="r", **kwargs)
    if exact_pcf is not None:
        axis.plot(
            pcf_dataframe["r"],
            exact_pcf(pcf_dataframe["r"]),
            "g",
            label="exact pcf",
        )
    plot_poisson(pcf_dataframe["r"], axis=axis, linestyle=(0, (5, 5)))

    axis.legend()
    axis.set_xlabel(r"Radius ($r$)")
    axis.set_ylabel(r"Pair correlation function ($g(r)$)")

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_sf_hankel_quadrature(
    norm_k,
    sf,
    axis,
    k_min,
    exact_sf,
    error_bar,
    file_name,
    label=r"$\widehat{S}_{H}$",
    **binning_params
):
    """Plot the approximations of the structure factor (results of :py:meth:`~structure_factor.hankel_quadrature`) with means and error bars over bins, see :py:meth:`~structure_factor.utils._bin_statistics`."""
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 5))

    plot_approximation(
        norm_k,
        sf,
        axis=axis,
        label=label,
        marker=".",
        linestyle="",
        color="grey",
        markersize=4,
    )
    if exact_sf is not None:
        plot_exact(norm_k, exact_sf, axis=axis, label=r"Exact $\mathcal{S}(k)$")
    if error_bar:
        plot_summary(norm_k, sf, axis=axis, **binning_params)
    plot_poisson(norm_k, axis=axis)
    if k_min is not None:
        sf_interpolate = interpolate.interp1d(
            norm_k, sf, axis=0, fill_value="extrapolate", kind="cubic"
        )
        axis.loglog(
            k_min,
            sf_interpolate(k_min),
            "ro",
            label=r"$k_{\min}$",
        )
    axis.legend()
    axis.set_xlabel(r"Wavenumber ($k$)")
    axis.set_ylabel(r"Structure factor ($\mathcal{S}(k)$)")

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis
