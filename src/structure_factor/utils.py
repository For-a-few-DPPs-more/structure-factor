#!/usr/bin/env python3
# coding=utf-8

import numpy as np

from mpmath import fp as mpm
from scipy.special import j0, j1, jv, jn_zeros, y0, y1, yv
from scipy import interpolate, stats
import matplotlib.pyplot as plt

# todo bien renomer les variables
# todo clean up the file: remove unused utility functions like get_x, roots etc


def cleaning_data(data):
    data_clean = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    return data_clean


def get_random_number_generator(seed):
    """Turn seed into a np.random.Generator instance"""
    if isinstance(seed, np.random.Generator):
        return seed
    if seed is None or isinstance(seed, (int, np.integer)):
        return np.random.default_rng(seed)
    raise TypeError(
        "seed must be None, an np.random.Generator or an integer (int, np.integer)"
    )
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


def roots(d, N):
    # first N Roots of the Bessel J_(d/2-1) functions divided by pi.
    return np.array([mpm.besseljzero(d / 2 - 1, i + 1) for i in range(N)]) / np.pi


def get_x(h, zeros):
    return np.pi * psi(h * zeros) / h


def weight(d, zeros):
    return bessel2(d / 2 - 1, np.pi * zeros) / bessel1(d / 2, np.pi * zeros)


def psi(t):
    # equation 5.1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    return t * np.tanh((0.5 * np.pi) * np.sinh(t))


def d_psi(t):
    # equation 5.1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    threshold = 3.5  # threshold outside of which psi' plateaus to -1, 1
    out = np.sign(t)
    mask = np.abs(t) < threshold
    x = t[mask]
    out[mask] = np.pi * x * np.cosh(x) + np.sinh(np.pi * np.sinh(x))
    out[mask] /= 1.0 + np.cosh(np.pi * np.sinh(x))
    return out


def integrate_with_abs_odd_monomial(f, nu=0, h=0.1, n=100, f_even=False):
    # Section 1 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    x = bessel1_zeros(nu, n)
    weights = bessel2(nu, x) / bessel1(nu + 1, x)  # equation 1.2
    x *= h / np.pi  # equivalent of xi variable
    # equation 1.1
    if f_even:
        return 2.0 * h * np.sum(weights * np.power(x, 2 * nu + 1) * f(x))
    return h * np.sum(weights * np.power(x, 2 * nu + 1) * (f(x) + f(-x)))


def integrate_with_bessel_function_half_line(f, nu=0, h=0.01, n=1000):
    # Section 5 Ogata https://www.kurims.kyoto-u.ac.jp/~okamoto/paper/Publ_RIMS_DE/41-4-40.pdf
    t = bessel1_zeros(nu, n)
    weights = bessel2(nu, t) / bessel1(nu + 1, t)  # equation 1.2
    t *= h / np.pi  # equivalent of xi variable
    # Change of variable equation 5.2
    x = (np.pi / h) * psi(t)
    return np.pi * np.sum(weights * f(x) * bessel1(nu, x) * d_psi(t))


def compute_scattering_intensity(k, data):
    X = data
    n = X.shape[0]
    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, X.T)), axis=1)))
    si /= n
    return si


def _binning_function(x_data, y_data, **binning_params):
    Xs = x_data.ravel()
    Ys = y_data.ravel()
    x2listy = {}
    for x, y in zip(Xs, Ys):
        try:
            x2listy[x].append(y)
        except KeyError:
            x2listy[x] = [y]

    x2meanY = {x: np.mean(x2listy[x]) for x in x2listy}
    x_meanY = sorted(x2meanY.items())
    mean_x, mean_y = zip(*x_meanY)
    bin_mean, bin_edges, binnumber = stats.binned_statistic(
        mean_x, mean_y, statistic="mean", **binning_params
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2

    bin_std, bin_edges, misc = stats.binned_statistic(
        mean_x, mean_y, statistic=np.std, **binning_params
    )
    return (bin_centers, bin_mean, bin_std)


def _lsf(x_data, y_data, stop=None):
    if stop is not None:
        x = x_data[:stop]
        y = y_data[:stop]
    else:
        x = x_data
        y = y_data

    N = x.shape[0]
    x_square = x ** 2
    xy = x * y
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    slope = (N * np.sum(xy) - sum_x * sum_y) / (N * np.sum(x_square) - sum_x ** 2)
    y_intercept = (sum_y - slope * sum_x) / N
    fitted_line = lambda t: slope * t + y_intercept
    return fitted_line


def plot_scattering_intensity_(
    points, norm_k, si, plot_type, exact_sf, error_bar, save, **binning_params
):
    r"""[summary]

    Args:
        points :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
        norm_k (np.array): output vector of the function ``compute_scattering_intensity``.
        si (n.array): output vector of the function ``compute_scattering_intensity``.
        plot_type  (str): ("plot", "color_level" and "all"), specify the type of the plot to be shown. Defaults to "plot".
        **binning_params: binning parameters used by ``stats.binned_statistic``, to find the mean of ``si``over subinternals of ``norm_k``for more details see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>. Note that the parameter ``statistic``is fixed to ``mean``.
    """

    bin_centers, bin_mean, bin_std = _binning_function(
        norm_k.ravel(), si.ravel(), **binning_params
    )

    if plot_type == "all":
        if len(norm_k.shape) < 2:
            raise ValueError(
                "the scattering intensity should be evaluated on a meshgrid or choose plot_type='plot'. "
            )
        else:
            log_si = np.log10(si)
            m, n = log_si.shape
            m /= 2
            n /= 2

            fig, ax = plt.subplots(1, 3, figsize=(24, 6))
            ax[0].plot(points[:, 0], points[:, 1], "b,")
            ax[0].title.set_text("Points configuration")
            ax[1].loglog(norm_k.ravel(), si.ravel(), "k,", marker=",")
            ax[1].loglog(bin_centers, bin_mean, "b.")
            ax[1].loglog(norm_k.ravel(), np.ones_like(norm_k).ravel(), "r--")
            if exact_sf is not None:
                ax[1].loglog(norm_k.ravel(), exact_sf(norm_k).ravel(), "g", zorder=5)
                ax[1].legend(
                    ["SI", "Mean(SI)", "theo", "Exact sf"],
                    shadow=True,
                    loc="lower right",
                )
            else:
                ax[1].legend(["SI", "Mean(SI)", "theo"], shadow=True, loc="lower right")
            ax[1].errorbar(
                bin_centers,
                bin_mean,
                yerr=bin_std,
                fmt="b",
                elinewidth=2,
                ecolor="r",
                capsize=3,
                capthick=1,
                zorder=4,
            )
            ax[1].set_xlabel("Wave length")
            ax[1].set_ylabel("Scattering intensity")
            ax[1].title.set_text("loglog plot")

            f_0 = ax[2].imshow(
                log_si,
                extent=[-n, n, -m, m],
                cmap="PRGn",
            )
            fig.colorbar(f_0, ax=ax[2])
            ax[2].title.set_text("scattering intensity")
            plt.show()
            if save:
                fig.savefig("si.pdf", bbox_inches="tight")
    elif plot_type == "plot":
        fig = plt.figure(figsize=(10, 7))
        plt.loglog(norm_k.ravel(), si.ravel(), "k,", zorder=1)
        plt.loglog(bin_centers, bin_mean, "b.", zorder=3)
        plt.loglog(norm_k.ravel(), np.ones_like(norm_k.ravel()), "r--", zorder=2)
        if error_bar:
            plt.errorbar(
                bin_centers,
                bin_mean,
                yerr=bin_std,
                fmt="b",
                elinewidth=2,
                ecolor="r",
                capsize=3,
                capthick=1,
                zorder=4,
            )
        if exact_sf is not None:
            plt.loglog(norm_k.ravel(), exact_sf(norm_k.ravel()), "g", zorder=5)
            plt.legend(
                ["SI", "Mean(SI)", "theo", "error bar", "Exact sf"], loc="lower right"
            )
        else:
            plt.legend(["SI", "Mean(SI)", "theo", "error bar"], loc="lower right")
        plt.xlabel("Wave length ")
        plt.ylabel("Scattering intensity")
        plt.title("loglog plot")
        plt.show()
        if save:
            fig.savefig("si_figure.pdf", bbox_inches="tight")

    elif plot_type == "color_level":
        print(len(norm_k.shape))
        if len(norm_k.shape) < 2:
            raise ValueError(
                "the scattering intensity should be evaluated on a meshgrid or choose plot_type = 'plot'. "
            )
        else:
            log_si = np.log10(si)
            m, n = log_si.shape
            m /= 2
            n /= 2
            f_0 = plt.imshow(
                log_si,
                extent=[-n, n, -m, m],
                cmap="PRGn",
            )
            plt.colorbar(f_0)
            plt.title("Scattering intensity")
            plt.show()
        if save:
            fig = f_0.get_figure()
            fig.savefig("si_figure.pdf", bbox_inches="tight")
    else:
        raise ValueError(
            "plot_type should be one of the following str: 'all', 'plot' and 'color_level'.  "
        )


def plot_pcf_(pcf_DataFrame, exact_pcf, save, **kwargs):
    ax = pcf_DataFrame.plot.line(x="r", **kwargs)
    if exact_pcf is not None:
        ax.plot(
            pcf_DataFrame["r"],
            exact_pcf(pcf_DataFrame["r"]),
            "r",
            label="exact pcf",
        )
        ax.legend()
    ax.set_xlabel("r")
    ax.set_ylabel("pcf")
    plt.show()
    if save:
        fig = ax.get_figure()
        fig.savefig("pcf_figure.pdf", bbox_inches="tight")


def plot_sf_via_hankel_(k, sf, k_min, exact_sf, error_bar, save, **binning_params):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(k, sf, "k.", label="approx sf")
    ax[0].plot(k, sf, "k")
    if exact_sf is not None:
        ax[0].plot(
            k,
            exact_sf(k),
            "g",
            label="exact sf",
        )
    if k_min is not None:
        sf_interpolate = interpolate.interp1d(
            k, sf, axis=0, fill_value="extrapolate", kind="cubic"
        )
        ax[0].plot(
            k_min,
            sf_interpolate(k_min),
            "ro",
            label="k_min",
        )
    if error_bar:
        bin_centers, bin_mean, bin_std = _binning_function(
            k.ravel(), sf.ravel(), **binning_params
        )
        ax[0].errorbar(
            bin_centers,
            bin_mean,
            yerr=bin_std,
            fmt="b",
            elinewidth=2,
            ecolor="r",
            capsize=3,
            capthick=1,
            zorder=4,
        )

    ax[0].plot(k, np.ones_like(k), "r--", label="theo")
    ax[0].legend()
    ax[0].set_xlabel("wave length")
    ax[0].set_ylabel("sf")
    ax[0].title.set_text("plot")

    ax[1].loglog(k, sf, "k.", label="approx sf")
    if exact_sf is not None:
        ax[1].loglog(
            k,
            exact_sf(k),
            "g",
            label="exact sf",
        )
    if k_min is not None:
        sf_interpolate = interpolate.interp1d(
            k, sf, axis=0, fill_value="extrapolate", kind="cubic"
        )
        ax[1].loglog(
            k_min,
            sf_interpolate(k_min),
            "ro",
            label="k_min",
        )
    ax[1].loglog(k, np.ones_like(k), "r--", label="theo")
    if error_bar:
        bin_centers, bin_mean, bin_std = _binning_function(
            k.ravel(), sf.ravel(), **binning_params
        )
        ax[1].errorbar(
            bin_centers,
            bin_mean,
            yerr=bin_std,
            fmt="b",
            elinewidth=2,
            ecolor="r",
            capsize=3,
            capthick=1,
            zorder=4,
        )

    ax[1].legend()
    ax[1].set_xlabel("wave length")
    ax[1].set_ylabel("sf")
    ax[1].title.set_text("loglog plot")
    plt.show()
    if save:
        fig.savefig("sf_via_hankel_figure.pdf", bbox_inches="tight")
