import matplotlib.pyplot as plt
import numpy as np

from structure_factor.utils import _bin_statistics, _sort_vectors

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
    rasterized=True,
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
