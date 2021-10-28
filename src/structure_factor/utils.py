#!/usr/bin/env python3
# coding=utf-8

# import pandas as pd
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, stats
from scipy.special import j0, j1, jn_zeros, jv, y0, y1, yv


def set_nan_inf_to_zero(array, nan=0, posinf=0, neginf=0):
    """Set nan, posinf and neginf values of ``array`` to 0."""
    return np.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf)


def get_random_number_generator(seed=None):
    """Turn seed into a np.random.Generator instance."""
    return np.random.default_rng(seed)


def bessel1(order, x):
    """Evaluate `first kind bessel function <https://en.wikipedia.org/wiki/Bessel_function>`_."""
    if order == 0:
        return j0(x)
    if order == 1:
        return j1(x)
    return jv(order, x)


def bessel1_zeros(order, nb_zeros):
    """Evaluate zeros of the `first kind bessel function <https://en.wikipedia.org/wiki/Bessel_function>`_."""
    return jn_zeros(order, nb_zeros)


def bessel2(order, x):
    """Evaluate `second kind bessel function <https://en.wikipedia.org/wiki/Bessel_function>`_."""
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


###### utils for the class StructureFactor

# ? see difference between odd and even L why the final size of the meshgrid change?


def allowed_wave_vectors(d, L, k_max, meshgrid_shape=None):
    r"""Given a realization of a point process in a cubic window with length :math:`L`, return a subset of the set of the 'allowed' wave vectors (defined below) :math:`\{\mathbf{k}_i\}_i` at which the structure factor :math:`S(\mathbf{k}_i)` is consistently estimated by the scattering intensity :math:`\widehat{S}_{SI}` .

    The allowed waves vectors are :math:`\{\mathbf{k}_i\}_i` s.t.,

    .. math::

        \{\mathbf{k}_i\}_i \subset \{\frac{2 \pi}{L} \mathbf{n} ~ ; ~ \mathbf{n} \in (\mathbb{Z}^d)^\ast \}.

    The maximum and the number of output allowed wave vectors are specified by setting the parameters `k_max` and `meshgrid_shape`.


    Args:

        d (int): dimension of the space containing the point process.

        L (float): length of the cubic window containing the sample of points.

        k_max (float): maximum component of the waves vectors i.e., for any output allowed wave vector :math:`\mathbf{k}=(k_1,...,k_d)`, we have :math:`k_i \leq k\_max` for all i. This implies that the maximum wave vectors will be :math:`(k\_max, ... k\_max)`.

        meshgrid_shape (tuple, optional): tuple of length `d`, where each element specify the number of component over the corresponding axis. It consists of the associated size of the meshgrid of allowed waves. For example if we are working in 2 dimensions, letting meshgid_shape=(2,3) will give a meshgrid of allowed waves formed by a vector of 2 values over the x-axis and a vectors of 3 values over the y-axis. Defaults to None.

    Returns:
        tuple (np.ndarray, list):
            - k : np.array with d columns where each row is an allowed wave vector.
            - K : list of meshgrid where the elements of the list corresponding to the 2D respresentation of the components of the wave vectors, i.e., it's a 2D representation of the vectors of allowed values ``k``. For example in dimension 2, if K =[X,Y] then X is the 2D representation of the x coordinates of the allowed wave vectors ``k`` i.e., the representation as meshgrid.

    """
    K = None
    n_max = np.floor(k_max * L / (2 * np.pi))  # maximum of ``n``

    if meshgrid_shape is None:
        warnings.warn(message="Taking all allowed wave vectors may be time consuming.")
        n_all = ()
        n_i = np.arange(-n_max, n_max + 1, step=1)
        n_i = n_i[n_i != 0]
        n_all = (n_i for i in range(0, d))
        X = np.meshgrid(*n_all, copy=False)
        T = []
        for i in range(0, d):
            T.append(X[i].ravel())
        n = np.column_stack(T)

    elif (np.array(meshgrid_shape) > (2 * n_max)).any():
        warnings.warn(
            message="meshgrid_shape should be less than the shape of meshgrid of the total allowed wave of points."
        )
        n_i = np.arange(-n_max, n_max + 1, step=1)
        n_all = ()
        n_i = np.arange(-n_max, n_max + 1, step=1)
        n_i = n_i[n_i != 0]
        n_all = (n_i for i in range(0, d))
        X = np.meshgrid(*n_all, copy=False)
        K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wave vectors
        # reshape allowed vectors as d columns
        T = []
        for i in range(0, d):
            T.append(X[i].ravel())
        n = np.column_stack(T)

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
            T = []
            # reshape allowed wave vector as q*d array
            for i in range(0, d):
                T.append(X[i].ravel())
            n = np.column_stack(T)  # allowed wave vectors  (d columns)

    k = 2 * np.pi * n / L
    return k, K


def compute_scattering_intensity(k, points):
    r"""Compute the scattering intensity which is an ensemble estimator of the structure factor of an ergodic stationary point process :math:`\mathcal{X} \subset \mathbb{R}^2`, defined below.

    .. math::
        SI(\mathbf{k}) = \left \lvert \sum_{x \in \mathcal{X}} \exp(- i \left\langle \mathbf{k}, \mathbf{x} \right\rangle) \right\rvert^2

    where :math:`\mathbf{k} \in \mathbb{R}^2` is a wave vector.

    Args:

        k (np.ndarray): np.array of d columns (where d is the dimesion of the space containing the points) where each row correspond to a wave vector. As mentioned before its recommended to keep the default ``k`` and to specify ``k_max`` instead, so that the approximation will be evaluated on allowed wavevectors. Defaults to None.

        points (np.ndarray): np.ndarray od d columns where each row consits a point from the realization of the point process.

    Returns:
        numpy.ndarray: Vector of evaluation of the scattering intensity on ``k``.

    .. seealso::

        `Wikipedia structure factor/scattering intensity <https://en.wikipedia.org/wiki/Structure_factor>`_.
    """
    n = points.shape[0]  # number of points
    if points.shape[1] != k.shape[1]:
        raise ValueError("k and points should have same number of columns")

    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, points.T)), axis=1)))
    si /= n

    # reshape the output

    return si


def _bin_statistics(x, y, **params):
    """Divide ``x`` into bins and evaluate the mean and the standard deviation of the corresponding element of ``y`` over the each bin.

    Args:
        x (np.ndarray): vector of data.
        y (np.ndarray): vector of data associated to the vector ``x``.

    Keyword args:
        params (dict): keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray):
            - ``bin_centers`` vector of centers of the bins associated to ``x``,
            - ``bin_mean`` vector of means of ``y`` over the bins,
            - ``std_mean`` vector of standard deviation of ``y`` over the bins.
    """
    bin_mean, bin_edges, _ = stats.binned_statistic(x, y, statistic="mean", **params)
    bin_std, _, _ = stats.binned_statistic(x, y, statistic="std", **params)
    count, _, _ = stats.binned_statistic(x, y, statistic="count", **params)
    bin_std = bin_std / np.sqrt(count)
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    return bin_centers, bin_mean, bin_std


def plot_poisson(x, axis, c="k", linestyle=(0, (5, 10)), label="Poisson"):
    r"""plot the pair correlation function :math:`g_{poisson}` and the structure factor :math:`S_{poisson}` corresponding to the Poisson point process.

    .. math::

        g_{poisson} = S_{poisson} = 1


    Args:
        x (np.array): x coordinate.

        axis (axis): axis on which to add the plot.

        c (str, optional): color of the plot. see `matplotlib <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ . Defaults to "k".

        linestyle (tuple, optional): linstyle of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_. Defaults to (0, (5, 10)).

        label (str, optional): specification of the label of the plot. Defaults to "Poisson".

    Returns:
        plot: plot of the pair correlation function and the structure factor of the Poisson point process over `x`.
    """
    axis.plot(x, np.ones_like(x), c=c, linestyle=linestyle, label=label)
    return axis


def plot_summary(x, y, axis, label=r"mean $\pm$ 3 $\cdot$ std", **binning_params):
    """Loglog plot the summary results of _bin_statistics function i.e. means and errors bars (3 standard deviations)."""
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


def plot_approximation(x, y, label, axis, color, linestyle, marker, markersize):
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
    """Loglog plot of the results of the scattering intensity :py:meth:`StructureFactor.scattering_intensity`, with the means and error bars over specific number of bins found via :py:meth:`~structure_factor.utils._bin_statistics`."""
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
    """Construct 3 subplots: point pattern, associated scattering intensity plot, associated scattering intensity color level."""
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
    """Plot approximation of structure factor using :py:meth:`~structure_factor.hankel_quadrature` with means and error bars over bins."""
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
