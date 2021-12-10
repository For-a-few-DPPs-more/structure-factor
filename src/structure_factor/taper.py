import numpy as np


def h0(points, window):
    """The indicator of a box window divided by the square root of the volume of the window.

    Args:
        points (np.ndarray): Points of a realization of a point process.
        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.

    Returns:
        [type]: [description]
    """
    window_volume = window.volume
    h0 = 1 / np.sqrt(window_volume)
    return h0


def ft_h0(k, window):
    r"""Fourier transform of h0 (the indicator of a box window divided by the square root of the volume of the window).

    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.

    Return:
        numpy.array: The evaluated Fourier transform on `k`.
    """
    window_bounds = window.bounds
    window_volume = window.volume
    s_k = 2 * np.sin(k * np.diff(window_bounds).T / 2) / k
    ft_h0 = np.prod(s_k, axis=1) / np.sqrt(window_volume)
    return ft_h0
