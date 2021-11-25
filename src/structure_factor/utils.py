#!/usr/bin/env python3
# coding=utf-8

# import pandas as pd
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, stats
from scipy.special import j0, j1, jn_zeros, jv, y0, y1, yv


def get_random_number_generator(seed=None):
    """Turn seed into a np.random.Generator instance."""
    return np.random.default_rng(seed)


# theoretical structure factors and pair correlation functions


def pair_correlation_function_ginibre(x):
    return 1.0 - np.exp(-(x ** 2))


def structure_factor_poisson(k):
    return np.ones_like(k)


def structure_factor_ginibre(k):
    return 1.0 - np.exp(-(k ** 2) / 4)


# utils for hyperuniformity.py


def _sort_vectors(k, x_k, y_k=None):
    """Sort ``k`` by increasing order and rearranging the associated vectors to ``k``, ``x_k``and ``y_k``.

    Args:
        k (np.array): Vector to be sorted by increasing order.
        x_k (np.array): Vector of evaluations associated with ``k``.
        y_k (np.array, optional): Vector of evaluations associated with ``k``. Defaults to None.

    Returns:
        (np.array, np.array, np.array): ``k`` sorted by increasing order and the associated vectors ``x_k``and ``y_k``.
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
        x (np.array): Vector of data.
        y (np.array): Vector of data associated with the vector ``x``.

    Keyword args:
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


def _compute_k_min(r_max, step_size):
    """Estimate a lower bound of the wavenumbers for which the approximation of the Hankel Transform by Ogata's quadrature is confident. See :cite:`Oga05`.

    Args:
        r_max (float): Maximum radius, on which the input function :math:`f` to be Hankel transformed, was evaluated before the interpolation.

        step_size (float): Stepsize used in the quadrature of Ogata.

    Returns:
        float: Wavenumbers lower bound's.
    """
    return (2.7 * np.pi) / (r_max * step_size)


# utils for structure_factor.py


def set_nan_inf_to_zero(array, nan=0, posinf=0, neginf=0):
    """Set nan, posinf, and neginf values of ``array`` to 0."""
    return np.nan_to_num(array, nan=nan, posinf=posinf, neginf=neginf)


def _reshape_meshgrid(X):
    r"""Reshape the list of meshgrids ``X`` as np.ndarray, where each column is associated to an element (meshgrid) of the list `X``.

    Args:
        X (list): List of meshgrids.

    Returns:
        n: np.ndarray where each meshgrid of the original list ``X`` is stacked as a column.
    """
    T = []
    d = len(X)
    for i in range(0, d):
        T.append(X[i].ravel())
    n = np.column_stack(T)  # stack in d columns
    return n


def allowed_wave_vectors(d, L, k_max, meshgrid_shape=None):
    r"""Return a subset of the d-dimensional allowed wave vectors corresponding to a cubic window of length ``L``.

    Args:

        d (int): Dimension of the space containing the point process.

        L (float): Length of the cubic window containing the point process realization.

        k_max (float): Supremum of the components of the allowed wavevectors on which the scattering intensity to be evaluated; i.e., for any allowed wavevector :math:`\mathbf{k}=(k_1,...,k_d)`, :math:`k_i \leq k\_max` for all i. This implies that the maximum of the output vector ``k_norm`` will be approximately equal to the norm of the vector :math:`(k\_max, ... k\_max)`.

        meshgrid_shape (tuple, optional): Tuple of length `d`, where each element specifies the number of components over an axis. These axes are crossed to form a subset of :math:`\mathbb{Z}^d` used to construct a set of allowed wavevectors. i.g., if d=2, setting meshgid_shape=(2,3) will construct a meshgrid of allowed wavevectors formed by a vector of 2 values over the x-axis and a vector of 3 values over the y-axis. Defaults to None, which will run the calculation over **all** the allowed wavevectors. Defaults to None.

    Returns:
        tuple (np.ndarray, list):
            - k : np.array with ``d`` columns where each row is an allowed wave vector.
            - K : List of meshgrids, (the elements of the list correspond to the 2D representation of the components of the wavevectors, i.e., a 2D representation of the vectors of allowed waves ``k``). i.g., in dimension 2, if K =[X,Y] then X is the 2D representation of the x coordinates of the allowed wavevectors ``k`` (representation as a meshgrid).

    .. proof:definition::

        The set of the allowed wavevectors :math:`\{\mathbf{k}_i\}_i` is defined by

        .. math::

            \{\mathbf{k}_i\}_i = \{\frac{2 \pi}{L} \mathbf{n} ~ ; ~ \mathbf{n} \in (\mathbb{Z}^d)^\ast \}.

        Note that the maximum ``n`` and the number of output allowed wavevectors returned by :py:meth:`allowed_wave_vectors`, are specified by the input parameters ``k_max`` and ``meshgrid_shape``.
    """
    K = None
    n_max = np.floor(k_max * L / (2 * np.pi))  # maximum of ``n``

    # warnings
    if meshgrid_shape is None:
        warnings.warn(
            message="The computation on all allowed wave vectors may be time-consuming."
        )
    elif (np.array(meshgrid_shape) > (2 * n_max)).any():
        warnings.warn(
            message="Each component of the argument 'meshgrid_shape' should be less than or equal to the cardinality of the (total) set of allowed wavevectors."
        )

    if meshgrid_shape is None or (np.array(meshgrid_shape) > (2 * n_max)).any():
        n_all = ()
        n_i = np.arange(-n_max, n_max + 1, step=1)
        n_i = n_i[n_i != 0]
        n_all = (n_i for i in range(0, d))
        X = np.meshgrid(*n_all, copy=False)
        K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wavevectors
        n = _reshape_meshgrid(X)  # reshape as d columns

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
            K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wavevectors
            n = _reshape_meshgrid(X)  # reshape as d columns

    k = 2 * np.pi * n / L
    return k, K


def compute_scattering_intensity(k, points):
    r"""Compute the scattering intensity of ``points`` at each wavevector in ``k``.

    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the scattering intensity is to be evaluated.

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


def plot_poisson(x, axis, c="k", linestyle=(0, (5, 10)), label="Poisson"):
    r"""Plot the pair correlation function :math:`g_{poisson}` and the structure factor :math:`S_{poisson}` corresponding to the Poisson point process.

    Args:
        x (np.array): x coordinate.

        axis (matplotlib.axis): Axis on which to add the plot.

        c (str, optional): Color of the plot. see `matplotlib <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ . Defaults to "k".

        linestyle (tuple, optional): Linstyle of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_. Defaults to (0, (5, 10)).

        label (regexp, optional): Label of the plot. Defaults to "Poisson".

    Returns:
        matplotlib.plot: Plot of the pair correlation function and the structure factor of the Poisson point process over ``x``.

    """
    axis.plot(x, np.ones_like(x), c=c, linestyle=linestyle, label=label)
    return axis


def plot_summary(x, y, axis, label=r"mean $\pm$ 3 $\cdot$ std", **binning_params):
    r"""Loglog plot the summary results of :py:func:`~structure_factor.utils._bin_statistics` i.e., means and errors bars (3 standard deviations).

    [extended_summary]

    Args:
        x (np.ndarray): x coordinate.
        y (np.ndarray): y coordinate.
        axis (matplotlib.axis): Axis on which to add the plot.
        label (regexp, optional):  Label of the plot. Defaults to r"mean $\pm$ 3 $\cdot$ std".

    Returns:
        matplotlib.plot: Plot of the results of :py:meth:`~structure_factor.utils._bin_statistics` applied on ``x`` and ``y`` .
    """
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
    r"""Loglog plot of a callable function ``y`` evaluated on the vector ``x``.

    Args:
        x (np.ndarray): x coordinate.
        y (callable): Function to evaluate on ``x``.
        axis (matplotlib.axis): Axis on which to add the plot.
        label (regexp, optional):  Label of the plot.

    Returns:
        matplotlib.plot: Plot of ``y`` with respect to ``x``.
    """
    axis.loglog(x, y(x), "g", label=label)
    return axis


def plot_approximation(x, y, axis, label, color, linestyle, marker, markersize):
    r"""Loglog plot of ``y`` w.r.t. ``x``.

    Args:
        x (np.ndarray): x coordinate.
        y (np.ndarray): y coordinate.
        axis (matplotlib.axis): Axis on which to add the plot.
        label (regexp, optional):  Label of the plot.
        color (matplotlib.color): Color of the plot. see `color <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ .
        linestyle (tuple): Style of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_.
        marker (matplotlib.marker): Marker of `marker <https://matplotlib.org/stable/api/markers_api.html>`_.
        markersize (float): Marker size.

    Returns:
        matplotlib.plot: Loglog plot of ``y`` w.r.t. ``x``
    """
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
    k_norm,
    si,
    axis=None,
    exact_sf=None,
    error_bar=False,
    file_name="",
    **binning_params
):
    r"""Loglog plot of the results of the scattering intensity :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, with the means and error bars over specific number of bins found via :py:func:`~structure_factor.utils._bin_statistics`.

    Args:

        k_norm (np.ndarray): Wavenumbers.

        si (np.ndarray): Scattering intensity corresponding to ``k_norm``.

        axis (matplotlib.axis, optional): Axis on which to add the plot. Defaults to None.

        exact_sf (callable, optional): Structure factor of the point process. Defaults to None.

        error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``si`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

        file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".
    """
    k_norm = k_norm.ravel()
    si = si.ravel()
    if axis is None:
        _, axis = plt.subplots(figsize=(8, 6))

    plot_poisson(k_norm, axis=axis)
    if exact_sf is not None:
        plot_exact(k_norm, exact_sf, axis=axis, label=r"Exact $S(k)$")

    plot_approximation(
        k_norm,
        si,
        axis=axis,
        label=r"$\widehat{S}_{SI}$",
        color="grey",
        linestyle="",
        marker=".",
        markersize=1.5,
    )

    if error_bar:
        plot_summary(k_norm, si, axis=axis, **binning_params)

    axis.set_xlabel(r"Wavenumber ($||\mathbf{k}||$)")
    axis.set_ylabel(r"Structure factor ($S(k)$)")
    axis.legend(loc=4, framealpha=0.2)

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_si_imshow(k_norm, si, axis, file_name):
    r"""Color level 2D plot, centered on zero.

    Args:
        k_norm (np.ndarray): Wavenumbers.
        si (np.ndarray): Scattering intensity corresponding to ``k_norm``.
        axis (matplotlib.axis): Axis on which to add the plot.
        file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".
    """
    if axis is None:
        _, axis = plt.subplots(figsize=(14, 8))
    if len(k_norm.shape) < 2:
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
    k_norm,
    si,
    exact_sf=None,
    error_bar=False,
    file_name="",
    window_res=None,
    **binning_params
):
    r"""Construct 3 subplots: point pattern, associated scattering intensity plot, associated scattering intensity color level (only for 2D point processes).

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type PointPattern containing a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``
        k_norm (np.ndarray): Wavenumbers.
        si (np.ndarray): Scattering intensity corresponding to ``k_norm``.
        exact_sf (callable, optional): Structure factor of the point process. Defaults to None.
        error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``si`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.
        file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".
        window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): New restriction window. It is useful when the sample of points is large, so for time and visualization purposes, it is better to restrict the plot of the point process to a smaller window. Defaults to None.
    """
    figure, axes = plt.subplots(1, 3, figsize=(24, 6))

    point_pattern.plot(axis=axes[0], window_res=window_res)
    plot_si_showcase(
        k_norm,
        si,
        axes[1],
        exact_sf,
        error_bar,
        file_name="",
        **binning_params,
    )
    plot_si_imshow(k_norm, si, axes[2], file_name="")

    if file_name:
        figure.savefig(file_name, bbox_inches="tight")

    return axes


def plot_pcf(pcf_dataframe, exact_pcf, file_name, **kwargs):
    r"""Plot the columns a DataFrame (excluding the first) with respect to the first columns.

    Args:
        pcf_dataframe (pandas.DataFrame): Output DataFrame of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.compute_pcf`.

        exact_pcf (callable): Function representing the theoretical pair correlation function of the point process.

        file_name (str): Name used to save the figure. The available output formats depend on the backend being used.

        Keyword Args:
            kwargs (dict): Keyword arguments of the function `pandas.DataFrame.plot.line <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html>`_.

    """
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
    k_norm,
    sf,
    axis,
    k_norm_min,
    exact_sf,
    error_bar,
    file_name,
    label=r"$\widehat{S}_{H}$",
    **binning_params
):
    r"""Plot the approximations of the structure factor (results of :py:meth:`~structure_factor.hankel_quadrature`) with means and error bars over bins, see :py:meth:`~structure_factor.utils._bin_statistics`.

    Args:

        k_norm (np.array): Vector of wavenumbers (i.e., norms of waves) on which the structure factor has been approximated.

        sf (np.array): Approximation of the structure factor corresponding to ``k_norm``.

        axis (matplotlib.axis): Support axis of the plots.

        k_norm_min (float): Estimated lower bound of the wavenumbers (only when ``sf`` was approximated using **Ogata quadrature**).

        exact_sf (callable): Theoretical structure factor of the point process.

        error_bar (bool): If ``True``, ``k_norm`` and correspondingly ``si`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

        file_name (str): Name used to save the figure. The available output formats depend on the backend being used.

        label (regexp, optional):  Label of the plot. Default to r"$\widehat{S}_{H}$".

    Keyword Args:
        binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.

    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 5))

    plot_approximation(
        k_norm,
        sf,
        axis=axis,
        label=label,
        marker=".",
        linestyle="",
        color="grey",
        markersize=4,
    )
    if exact_sf is not None:
        plot_exact(k_norm, exact_sf, axis=axis, label=r"Exact $\mathcal{S}(k)$")
    if error_bar:
        plot_summary(k_norm, sf, axis=axis, **binning_params)
    plot_poisson(k_norm, axis=axis)
    if k_norm_min is not None:
        sf_interpolate = interpolate.interp1d(
            k_norm, sf, axis=0, fill_value="extrapolate", kind="cubic"
        )
        axis.loglog(
            k_norm_min,
            sf_interpolate(k_norm_min),
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
