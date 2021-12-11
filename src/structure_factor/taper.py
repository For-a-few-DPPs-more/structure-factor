import numpy as np


def h0(points, window):
    """The indicator of a box window divided by the square root of the volume of the window.

    Args:
        points (np.ndarray): Points of a realization of a point process.
        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.

    Returns:
        [type]: [description]
    """
    h = window.indicator_function(points).astype(float)
    h /= np.sqrt(window.volume)
    return h


def ft_h0(k, window):
    r"""Fourier transform of h0 (the indicator of a box window divided by the square root of the volume of the window).

    Args:

        k (np.ndarray): np.ndarray of d columns (where d is the dimension of the space containing ``points``). Each row is a wave vector on which the spectral estimator is to be evaluated.

        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.

    Return:
        numpy.array: The evaluated Fourier transform on `k`.
    """
    widths = 0.5 * np.diff(window.bounds.T, axis=0)
    sines = 2.0 * np.sin(k * widths) / k
    ft = np.prod(sines, axis=1)
    ft /= np.sqrt(window.volume)
    return ft
