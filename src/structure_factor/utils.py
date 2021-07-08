#!/usr/bin/env python3
# coding=utf-8

import numpy as np

from mpmath import fp as mpm
from scipy import interpolate, stats
import matplotlib.pyplot as plt

# todo bien renomer les variables
# todo clean up the file: remove unused utility functions like get_x, roots etc


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


def compute_scattering_intensity(k, data):
    X = data
    n = X.shape[0]
    si = np.square(np.abs(np.sum(np.exp(-1j * np.dot(k, X.T)), axis=1)))
    si /= n
    return si


def cleaning_data(data):
    data_clean = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
    return data_clean


def interpolate_correlation_function(self, r_vector, data_g):
    """given evaluations of the pair correlation function (g), it returns an interpolation of the total correlation function (h=g-1)

    Args:
        r_vector (np.array): vector containing the radius on which the pair correlation function is evaluated.
        data_g (np.array_like(r_vector)): vector containing the evaluations of the pair correlation function on r_vec.
    """

    return interpolate.interp1d(
        r_vector, data_g - 1.0, axis=0, fill_value="extrapolate", kind="cubic"
    )


def plot_scattering_intensity_(
    points, wave_length, si, plot_type, exact_sf=None, **binning_params
):
    r"""[summary]

    Args:
        points :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
        wave_length (np.array): output vector of the function ``compute_scattering_intensity``.
        si (n.array): output vector of the function ``compute_scattering_intensity``.
        plot_type  (str): ("plot", "color_level" and "all"), specify the type of the plot to be shown. Defaults to "plot".
        **binning_params: binning parameters used by ``stats.binned_statistic``, to find the mean of ``si``over subinternals of ``wave_length``for more details see <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>. Note that the parameter ``statistic``is fixed to ``mean``.
    """
    # todo add confidence interval over the bins
    bin_means, bin_edges, binnumber = stats.binned_statistic(
        wave_length.ravel(), si.ravel(), statistic="mean", **binning_params
    )
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[1:] - bin_width / 2
    if plot_type == "all":
        if len(wave_length.shape) < 2:
            raise ValueError(
                "the scattering intensity should be evaluated on a meshgrid or choose plot_type='plot'. "
            )
        else:
            log_si = np.log10(si)
            m, n = log_si.shape
            m /= 2
            n /= 2

            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
            ax[0].plot(points[:, 0], points[:, 1], "b,")
            ax[0].title.set_text("Points configuration")
            ax[1].loglog(wave_length, si, "k,")
            ax[1].loglog(bin_centers, bin_means, "b.")
            ax[1].loglog(wave_length, np.ones_like(wave_length), "r--")
            if exact_sf is not None:
                ax[1].loglog(wave_length, exact_sf(wave_length), "r", label="exact sf")
                ax[1].legend(
                    ["SI", "Mean(SI)", "y=1", "Exact sf"],
                    shadow=True,
                    loc="lower right",
                )
            else:
                ax[1].legend(["SI", "Mean(SI)", "y=1"], shadow=True, loc="lower right")
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
    elif plot_type == "plot":
        plt.loglog(wave_length, si, "k,")
        plt.loglog(bin_centers, bin_means, "b.")
        plt.loglog(wave_length, np.ones_like(wave_length), "r--")
        if exact_sf is not None:
            plt.loglog(wave_length, exact_sf(wave_length), "r")
            plt.legend(["SI", "Mean(SI)", "y=1", "exact sf"], loc="lower right")
        else:
            plt.legend(["SI", "Mean(SI)", "y=1"], loc="lower right")
        plt.xlabel("Wave length ")
        plt.ylabel("Scattering intensity")
        plt.title("loglog plot")
        plt.show()
    elif plot_type == "color_level":
        print(len(wave_length.shape))
        if len(wave_length.shape) < 2:
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
    else:
        raise ValueError(
            "plot_type should be one of the following str: 'all', 'plot' and 'color_level'.  "
        )


def plot_pcf_(pcf_DataFrame, exact_pcf=None, **kwargs):
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
