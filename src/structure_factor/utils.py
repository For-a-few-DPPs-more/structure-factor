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


# ? see difference between odd and even L why the final size of the meshgrid change?
def allowed_wave_values(L, max_k, meshgrid_size, max_add_k=1):
    r"""Given a realization of a point process in a cubic window with length :math:`L`, compute the 'allowed' wave vectors :math:`(k_i)` at which the structure factor :math:`S(k_i)` is consistently estimated by the scattering intensity defined below.

    .. math::

        \{\frac{2 \pi}{L} \mathbf{n} ~ ; ~ \mathbf{n} \in (\mathbb{Z}^d)^\ast, \left\lVert \mathbf{n} \right\rVert \leq \text{ max_k}\}

    # todo add bibliographic reference

    Args:
        L (float): Length of the cubic window.

        max_k (float): Maximum norm of the wave vectors.

        # todo give clearer description of meshgrid_size
        meshgrid_size (int): Size of the meshgrid of allowed values if ``k_vector`` is set to None and ``max_k`` is specified. **Warning:** setting big value in ``meshgrid_size`` could be time consuming when the sample has a lot of points.

        # todo give clearer description of max_add_k
        max_add_k (float): Maximum component of the allowed wave vectors to be added. In other words, in the case of the evaluation on a vector of allowed values (without specifying ``meshgrid_size``),  ``max_add_k`` can be used to add allowed values in a certain region for better precision. **Warning:** setting big value in ``max_add_k`` could be time consuming when the sample has a lot of points. Defaults to 1.

    Returns:
        numpy.ndarray: array of sizw :math:`N \times d` collecting the 'allowed' wave vectors.
    """
    max_n = np.floor(max_k * L / (2 * np.pi))  # maximum of ``k_vector``
    if meshgrid_size is None:  # Add extra allowed values near zero
        n_vector = np.linspace(1, max_n, int(max_n))
        k_vector = 2 * np.pi * np.column_stack((n_vector, n_vector)) / L
        max_add_n = np.floor(max_add_k * L / (2 * np.pi))
        add_n_vector = np.linspace(1, np.int(max_add_n), np.int(max_add_n))
        X, Y = np.meshgrid(add_n_vector, add_n_vector)
        add_k_vector = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L
        k_vector = np.concatenate((add_k_vector, k_vector))

    elif meshgrid_size > (2 * max_n):
        warnings.warn(
            message="meshgrid_size should be less than the total allowed number of points.",
            category=DeprecationWarning,
        )
        n_vector = np.arange(-max_n, max_n + 1, step=1)

    else:
        n_vector = np.linspace(
            -max_n, max_n, num=meshgrid_size, dtype=int, endpoint=True
        )
        if np.count_nonzero(n_vector == 0) != 0:
            n_vector = np.linspace(
                -max_n, max_n, num=meshgrid_size + 1, dtype=int, endpoint=True
            )

    n_vector = n_vector[n_vector != 0]
    X, Y = np.meshgrid(n_vector, n_vector)
    k_vector = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L

    return k_vector


def compute_scattering_intensity(k, points):
    r"""Compute the scattering intensity which is an ensemble estimator of the structure factor of an ergodic stationary point process :math:`\mathcal{X} \subset \mathbb{R}^2`, defined below.

    .. math::
        SI(\mathbf{k}) = \left \lvert \sum_{x \in \mathcal{X}} \exp(- i \left\langle \mathbf{k}, \mathbf{x} \right\rangle) \right\rvert^2

    where :math:`\mathbf{k} \in \mathbb{R}^2` is a wave vector.

    Args:
        k (numpy.ndarray): array of size :math:`N_1 \times 2` containing :math:`N_1` two dimensional wave vectors.
        points (numpy.ndarray): array of size :math:`N_2 \times 2` containing the :math:`N_2` points of a realization of the point process :math:`\mathcal{X}`.

    Returns:
        numpy.ndarray: Vector of evaluation of the scattering intensity on ``k``.

    .. seealso::

        `Wikipedia structure factor/scattering intensity <https://en.wikipedia.org/wiki/Structure_factor>`_.
    """
    n = points.shape[0]
    if points.shape[1] != k.shape[1]:
        raise ValueError("k and points should have same number of columns")

    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, points.T)), axis=1)))
    si /= n
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
    # todo add docstring
    axis.plot(x, np.ones_like(x), c=c, linestyle=linestyle, label=label)
    return axis


def plot_summary(x, y, axis, label="$mean \pm 3 \cdot std$", **binning_params):
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
    """Loglog plot of the results of the scattering intensity :py:meth:`StructureFactor.compute_sf_scattering_intensity`, with the means and error bars over specific number of bins found via :py:meth:`~structure_factor.utils._bin_statistics`."""
    norm_k = norm_k.ravel()
    si = si.ravel()
    if axis is None:
        _, axis = plt.subplots(figsize=(8, 6))

    plot_poisson(norm_k, axis=axis)
    if exact_sf is not None:
        plot_exact(norm_k, exact_sf, axis=axis, label="Exact $S(k)$")

    plot_approximation(
        norm_k,
        si,
        axis=axis,
        label="$\widehat{S}_{SI}$",
        color="grey",
        linestyle="",
        marker=".",
        markersize=1.5,
    )

    if error_bar:
        plot_summary(norm_k, si, axis=axis, **binning_params)

    axis.set_xlabel("Wavenumber ($||\mathbf{k}||$)")
    axis.set_ylabel("Structure factor ($S(k)$)")
    axis.legend(loc=4, framealpha=0.2)

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_si_imshow(norm_k, si, axis, file_name):
    """Color level plot, centered on zero."""
    if len(norm_k.shape) < 2:
        raise ValueError(
            "the scattering intensity should be evaluated on a meshgrid or choose plot_type = 'plot'. "
        )
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
    axis.set_xlabel("Radius ($r$)")
    axis.set_ylabel("Pair correlation function ($g(r)$)")

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
    label="$\widehat{S}_{H}$",
    **binning_params
):
    """Plot approximation of structure factor using :py:meth:`~structure_factor.compute_sf_hankel_quadrature` with means and error bars over bins."""
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
        plot_exact(norm_k, exact_sf, axis=axis, label="Exact $\mathcal{S}(k)$")
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
            label="k_min",
        )
    axis.legend()
    axis.set_xlabel("Wavenumber (k)")
    axis.set_ylabel("Structure factor ($\mathcal{S}(k)$)")

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis
