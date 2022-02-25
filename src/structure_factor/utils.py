"""Collection of secondary functions used in the principal modules."""

import warnings
from itertools import product

import matplotlib.pyplot as plt
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


def _reshape_meshgrid(X):
    r"""Reshape the list of meshgrids ``X`` as np.ndarray, where each column is associated to an element (meshgrid) of the list `X``.

    Args:
        X (list): List of meshgrids.

    Returns:
        np.ndarray: where each meshgrid of the original list ``X`` is stacked as a column.
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

    meshgrid_shape = np.fmin(meshgrid_shape, 2 * n_max)
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
            X = np.meshgrid(*ranges, copy=False)
            # K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wavevectors
            n = _reshape_meshgrid(X)  # reshape as d columns

        else:
            n_all = []
            i = 0
            for s in meshgrid_shape:
                n_i = np.linspace(-n_max[i], n_max[i], num=s, dtype=int, endpoint=True)
                if np.count_nonzero(n_i == 0) != 0:
                    n_i = np.linspace(
                        -n_max[i], n_max[i], num=s + 1, dtype=int, endpoint=True
                    )
                i += 1
                n_i = n_i[n_i != 0]
                n_all.append(n_i)

            X = np.meshgrid(*n_all, copy=False)
            # K = [X_i * 2 * np.pi / L for X_i in X]  # meshgrid of allowed wavevectors
            n = _reshape_meshgrid(X)  # reshape as d columns

        k = 2 * np.pi * n / L.T
    return k


# plot functions


def plot_poisson(x, axis, c="k", linestyle=(0, (5, 10)), label="Poisson"):
    r"""Plot the pair correlation function :math:`g_{poisson}` and the structure factor :math:`S_{poisson}` corresponding to the Poisson point process.

    Args:
        x (numpy.ndarray): x coordinate.

        axis (plt.Axes): Axis on which to add the plot.

        c (str, optional): Color of the plot. see `matplotlib <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ . Defaults to "k".

        linestyle (tuple, optional): Linstyle of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_. Defaults to (0, (5, 10)).

        label (regexp, optional): Label of the plot. Defaults to "Poisson".

    Returns:
        plt.Axes: Plot of the pair correlation function and the structure factor of the Poisson point process over ``x``.
    """
    axis.plot(x, np.ones_like(x), c=c, linestyle=linestyle, label=label)
    return axis


def plot_summary(
    x,
    y,
    axis,
    scale="log",
    label=r"mean $\pm$ 3 $\cdot$ std",
    fmt="b",
    ecolor="r",
    **binning_params
):
    r"""Loglog plot the summary results of :py:func:`~structure_factor.utils._bin_statistics` i.e., means and errors bars (3 standard deviations).

    Args:
        x (numpy.ndarray): x coordinate.

        y (numpy.ndarray): y coordinate.

        axis (plt.Axes): Axis on which to add the plot.

        label (regexp, optional):  Label of the plot. Defaults to r"mean $\pm$ 3 $\cdot$ std".

    Returns:
        plt.Axes: Plot of the results of :py:meth:`~structure_factor.utils._bin_statistics` applied on ``x`` and ``y`` .
    """
    bin_centers, bin_mean, bin_std = _bin_statistics(x, y, **binning_params)
    axis.plot(bin_centers, bin_mean, "b.")
    axis.errorbar(
        bin_centers,
        bin_mean,
        yerr=3 * bin_std,  # 3 times the standard deviation
        fmt=fmt,
        lw=1,
        ecolor=ecolor,
        capsize=3,
        capthick=1,
        label=label,
        zorder=4,
    )
    axis.legend(loc=4, framealpha=0.2)
    axis.set_yscale(scale)
    axis.set_xscale(scale)
    return axis


def plot_exact(x, y, axis, label):
    r"""Loglog plot of a callable function ``y`` evaluated on the vector ``x``.

    Args:
        x (numpy.ndarray): x coordinate.

        y (numpy.ndarray): y coordinate.

        axis (plt.Axes): Axis on which to add the plot.

        label (regexp, optional):  Label of the plot.

    Returns:
        plt.Axes: Plot of ``y`` with respect to ``x``.
    """
    x, y, _ = _sort_vectors(x, y)
    axis.plot(x, y, "g", label=label)
    return axis


def plot_approximation(
    x, y, axis, rasterized, label, color, linestyle, marker, markersize, scale="log"
):
    r"""Loglog plot of ``y`` w.r.t. ``x``.

    Args:
        x (numpy.ndarray): x coordinate.

        y (numpy.ndarray): y coordinate.

        axis (plt.Axes): Axis on which to add the plot.

        rasterized (bool): Rasterized option of `matlplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#:~:text=float-,rasterized,-bool>`_.

        label (regexp, optional):  Label of the plot.

        color (matplotlib.color): Color of the plot. see `color <https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.plot.html>`_ .

        linestyle (tuple): Style of the plot. see `linestyle <https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html>`_.

        marker (matplotlib.marker): Marker of `marker <https://matplotlib.org/stable/api/markers_api.html>`_.
        markersize (float): Marker size.

        scale(str, optional): Trigger between plot scales of `plt.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Defaults to `log`.

    Returns:
        plt.Axes: Loglog plot of ``y`` w.r.t. ``x``
    """
    axis.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        label=label,
        markersize=markersize,
        rasterized=rasterized,
    )
    axis.set_yscale(scale)
    axis.set_xscale(scale)
    return axis


def plot_estimation_showcase(
    k_norm,
    estimation,
    axis=None,
    scale="log",
    exact_sf=None,
    error_bar=False,
    label=r"$\widehat{S}$",
    rasterized="True",
    file_name="",
    **binning_params
):
    r"""Loglog plot of the results of the scattering intensity :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, with the means and error bars over specific number of bins found via :py:func:`~structure_factor.utils._bin_statistics`.

    Args:
        k_norm (numpy.ndarray): Wavenumbers.

        estimation (numpy.ndarray): Scattering intensity corresponding to ``k_norm``.

        axis (plt.Axes, optional): Axis on which to add the plot. Defaults to None.

        scale(str, optional): Trigger between plot scales of `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Defaults to `log`.


        exact_sf (callable, optional): Structure factor of the point process. Defaults to None.

        error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``estimation`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

        rasterized (bool, optional): Rasterized option of `matlplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#:~:text=float-,rasterized,-bool>`_. Defaults to True.

        file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".
    """
    k_norm = k_norm.ravel()
    estimation = estimation.ravel()
    if axis is None:
        _, axis = plt.subplots(figsize=(8, 6))
    plot_approximation(
        k_norm,
        estimation,
        axis=axis,
        label=label,
        color="grey",
        linestyle="",
        marker=".",
        markersize=1.5,
        rasterized=rasterized,
        scale=scale,
    )
    plot_poisson(k_norm, axis=axis)

    if error_bar:
        plot_summary(k_norm, estimation, axis=axis, scale=scale, **binning_params)

    if exact_sf is not None:
        plot_exact(k_norm, exact_sf(k_norm), axis=axis, label=r"Exact $S(\mathbf{k})$")

    axis.set_xlabel(r"Wavenumber ($||\mathbf{k}||$)")
    axis.set_ylabel(r"Structure factor ($S(\mathbf{k})$)")
    axis.legend(loc=4, framealpha=0.2)

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_estimation_imshow(k_norm, si, axis, file_name):
    r"""Color level 2D plot, centered on zero.

    Args:
        k_norm (numpy.ndarray): Wavenumbers.

        si (numpy.ndarray): Scattering intensity corresponding to ``k_norm``.

        axis (plt.Axes): Axis on which to add the plot.

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


def plot_estimation_all(
    point_pattern,
    k_norm,
    estimation,
    exact_sf=None,
    error_bar=False,
    label=r"$\widehat{S}$",
    rasterized=True,
    file_name="",
    window_res=None,
    scale="log",
    **binning_params
):
    r"""Construct 3 subplots: point pattern, associated scattering intensity plot, associated scattering intensity color level (only for 2D point processes).

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type PointPattern containing a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.

        k_norm (numpy.ndarray): Wavenumbers.

        estimation (numpy.ndarray): Scattering intensity corresponding to ``k_norm``.

        exact_sf (callable, optional): Structure factor of the point process. Defaults to None.

        error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``estimation`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

        rasterized (bool, optional): Rasterized option of `matlplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#:~:text=float-,rasterized,-bool>`_. Defaults to True.

        file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): New restriction window. It is useful when the sample of points is large, so for time and visualization purposes, it is better to restrict the plot of the point process to a smaller window. Defaults to None.

        scale(str, optional): Trigger between plot scales of `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Defaults to `log`.
    """
    figure, axes = plt.subplots(1, 3, figsize=(24, 6))

    point_pattern.plot(axis=axes[0], window=window_res)
    plot_estimation_showcase(
        k_norm,
        estimation,
        axis=axes[1],
        exact_sf=exact_sf,
        error_bar=error_bar,
        label=label,
        rasterized=rasterized,
        file_name="",
        scale=scale,
        **binning_params,
    )
    plot_estimation_imshow(k_norm, estimation, axes[2], file_name="")

    if file_name:
        figure.savefig(file_name, bbox_inches="tight")

    return axes


def plot_sf_hankel_quadrature(
    k_norm,
    estimation,
    axis,
    scale,
    k_norm_min,
    exact_sf,
    color,
    error_bar,
    label,
    file_name,
    **binning_params
):
    r"""Plot the approximations of the structure factor (results of :py:meth:`~structure_factor.structure_factor.StructureFactor.quadrature_estimator_isotropic`) with means and error bars over bins, see :py:meth:`~structure_factor.utils._bin_statistics`.

    Args:
        k_norm (numpy.ndarray): Vector of wavenumbers (i.e., norms of waves) on which the structure factor has been approximated.

        estimation (numpy.ndarray): Approximation of the structure factor corresponding to ``k_norm``.

        axis (plt.Axes): Support axis of the plots.

        scale(str): Trigger between plot scales of `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_.

        k_norm_min (float): Estimated lower bound of the wavenumbers (only when ``estimation`` was approximated using **Ogata quadrature**).

        exact_sf (callable): Theoretical structure factor of the point process.

        error_bar (bool): If ``True``, ``k_norm`` and correspondingly ``si`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

        file_name (str): Name used to save the figure. The available output formats depend on the backend being used.

        label (regexp):  Label of the plot.

    Keyword Args:
        binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.
    """
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 5))

    plot_approximation(
        k_norm,
        estimation,
        axis=axis,
        label=label,
        marker=".",
        linestyle="",
        color=color,
        markersize=4,
        scale=scale,
        rasterized=False,
    )
    if exact_sf is not None:
        plot_exact(k_norm, exact_sf(k_norm), axis=axis, label=r"Exact $S(k)$")
    if error_bar:
        plot_summary(k_norm, estimation, axis=axis, scale=scale, **binning_params)
    plot_poisson(k_norm, axis=axis)
    if k_norm_min is not None:
        sf_interpolate = interpolate.interp1d(
            k_norm, estimation, axis=0, fill_value="extrapolate", kind="cubic"
        )
        axis.loglog(
            k_norm_min,
            sf_interpolate(k_norm_min),
            "ro",
            label=r"$k_{min}$",
        )
    axis.legend()
    axis.set_xlabel(r"Wavenumber ($k$)")
    axis.set_ylabel(r"Structure factor ($S(k)$)")
    plt.show()
    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis
