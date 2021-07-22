#!/usr/bin/env python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jv, jn_zeros, y0, y1, yv
from scipy import interpolate, stats
import pandas as pd


def cleaning_data(data):
    data_clean = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    return data_clean


def get_random_number_generator(seed):
    """Turn seed into a np.random.Generator instance"""
    return np.random.default_rng(seed)


def bessel1(order, x):
    if order == 0:
        return j0(x)
    if order == 1:
        return j1(x)
    return jv(order, x)


def bessel1_zeros(order, nb_zeros):
    return jn_zeros(order, nb_zeros)


def bessel2(order, x):
    if order == 0:
        return y0(x)
    if order == 1:
        return y1(x)
    return yv(order, x)


def compute_scattering_intensity(k, data):
    X = data
    n = X.shape[0]
    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, X.T)), axis=1)))
    si /= n
    return si


#! touver un nom
def _binning_function(x, y, **params):
    df = pd.DataFrame({"x": x, "y": y})
    df = df.groupby("x").mean()
    x_unique, y_mean = df.index.to_numpy(), df["y"].to_numpy()

    bin_mean, bin_edges, _ = stats.binned_statistic(
        x_unique, y_mean, statistic="mean", **params
    )
    bin_std, bin_edges, _ = stats.binned_statistic(
        x_unique, y_mean, statistic="std", **params
    )
    bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

    return bin_centers, bin_mean, bin_std


def plot_summary(x, y, axis, label="Mean", **binning_params):
    bin_centers, bin_mean, bin_std = _binning_function(x, y, **binning_params)
    axis.loglog(bin_centers, bin_mean, "b.", label=label)
    axis.errorbar(
        bin_centers,
        bin_mean,
        yerr=bin_std,
        fmt="b",
        elinewidth=2,
        ecolor="r",
        capsize=3,
        capthick=1,
        label="Error bar",
        zorder=4,
    )
    return axis


def plot_exact(x, y, axis, label="Exact sf"):
    axis.loglog(x, y(x), "g", label=label)
    return axis


def plot_approximation(x, y, label="si(k)", axis=None, c="k,"):
    axis.loglog(x, y, c, label=label)
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
    # ? why .ravel()?
    norm_k = norm_k.ravel()
    si = si.ravel()
    if axis is None:
        _, axis = plt.subplots(figsize=(8, 6))

    axis.loglog(norm_k, np.ones_like(norm_k), "r--", label="Theo")
    plot_approximation(norm_k, si, axis=axis)
    if exact_sf is not None:
        plot_exact(norm_k, exact_sf, axis=axis)
    if error_bar:
        plot_summary(norm_k, si, axis=axis, **binning_params)

    axis.title.set_text("loglog plot")
    axis.set_xlabel("Wave length")
    axis.set_ylabel("Scattering intensity")
    axis.legend()

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_si_imshow(norm_k, si, axis, file_name):
    if len(norm_k.shape) < 2:
        raise ValueError(
            "the scattering intensity should be evaluated on a meshgrid or choose plot_type = 'plot'. "
        )
    if axis is None:
        _, axis = plt.subplots(figsize=(6, 6))
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
        axis.title.set_text("Scattering intensity")

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
    **binning_params
):
    figure, axes = plt.subplots(1, 3, figsize=(24, 6))

    point_pattern.plot(axis=axes[0])
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
    axis = pcf_dataframe.plot.line(x="r", **kwargs)
    if exact_pcf is not None:
        axis.plot(
            pcf_dataframe["r"],
            exact_pcf(pcf_dataframe["r"]),
            "r",
            label="exact pcf",
        )
    axis.legend()
    axis.set_xlabel("r")
    axis.set_ylabel("pcf")

    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def plot_sf_hankel_quadrature(
    norm_k, sf, axis, k_min, exact_sf, error_bar, file_name, **binning_params
):
    if axis is None:
        fig, axis = plt.subplots(figsize=(8, 5))

    plot_approximation(norm_k, sf, axis=axis, label="approx sf", c="k.")
    if exact_sf is not None:
        plot_exact(norm_k, exact_sf, axis=axis, label="exact sf")
    if error_bar:
        plot_summary(norm_k, sf, axis=axis, **binning_params)
    axis.plot(norm_k, np.ones_like(norm_k), "r--", label="Theo")
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
    axis.set_xlabel("wave length")
    axis.set_ylabel("sf")
    axis.title.set_text("loglog plot")

    if file_name:
        fig.savefig(file_name, bbox_inches="tight")
    return axis
