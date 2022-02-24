r"""Collection of classes representing tapers (also called tapering functions or window functions). Such tapers must have two methods.

- ``.taper(x, window)`` corresponding to the tapering function :math:`t(x, W)`,

- ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the tapering function.

These tapers satisfy some conditions listed in :cite:`HGBLR:22` (Sections 3.1, 4.3).

Example:
    .. literalinclude:: code/structure_factor/taper.py
        :language: python
"""

from itertools import product

import numpy as np

from structure_factor.spatial_windows import BoxWindow


class BartlettTaper:
    """Class representing the Bartlett tapering function."""

    @staticmethod
    def taper(x, window):
        r"""Evaluate the Bartlett taper :math:`t(x, W)` at ``x`` given the rectangular ``window`` :math:`W`.

        .. math::

            t(x, W) = \frac{1}{\sqrt{|W|}} 1_{x \in W}.

        Args:
            x (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the tapering function is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Returns:
            numpy.ndarray: evaluation of the taper :math:`t(x, W)`.
        """
        assert isinstance(window, BoxWindow)
        t = window.indicator_function(x).astype(float)
        t /= np.sqrt(window.volume)
        return t

    @staticmethod
    def ft_taper(k, window):
        r"""Evaluate the Fourier transform :math:`F[t(\cdot, W)](k)` of the taper :math:`t` (:py:meth:`~structure_factor.tapers.BartlettTaper.taper`).

        Args:
            k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the Fourier transform is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Return:
            numpy.ndarray: Evaluation of the Fourier transform at ``k``.
        """
        assert isinstance(window, BoxWindow)
        widths = 0.5 * np.diff(window.bounds.T, axis=0)

        mask = k == 0
        if np.any(mask):
            sines = np.zeros_like(k)
            widths_matrix = np.ones_like(k) * widths
            sines[mask] = 2.0 * widths_matrix[mask]
            np.logical_not(mask, out=mask)
            sines[mask] = 2.0 * np.sin(k[mask] * widths_matrix[mask]) / k[mask]
        else:
            sines = 2.0 * np.sin(k * widths) / k

        ft = np.prod(sines, axis=1)
        return ft / np.sqrt(window.volume)


class SineTaper:
    r"""Class representing the sine tapering function."""

    def __init__(self, p):
        self.p = np.array(p)

    def taper(self, x, window):
        r"""Evalute the sine taper :math:`t(x, W)` indexed by :py:attr:`~structure_factor.tapers.SineTaper.p` at ``x`` given the rectangular ``window`` :math:`W`.

        Args:
            x (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the tapering function is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Returns:
            numpy.ndarray: evaluation of the taper :math:`t(x, W)`.
        """
        assert isinstance(window, BoxWindow)
        widths = np.diff(window.bounds.T, axis=0)
        # d = x.shape[1]
        sines = x / widths + 0.5
        sines *= np.pi * self.p
        np.sin(sines, out=sines)

        t = window.indicator_function(x).astype(float)
        t *= np.prod(sines * np.sqrt(2), axis=1)
        t *= np.sqrt(1 / window.volume)  # normalization
        return t

    def ft_taper(self, k, window):
        r"""Evaluate the Fourier transform :math:`F[t(\cdot, W)](k)` of the taper :math:`t` (:py:meth:`~structure_factor.tapers.SineTaper.taper`).

        Args:
            k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the Fourier transform is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Return:
            numpy.ndarray: Evaluation of the Fourier transform at ``k``.
        """
        assert isinstance(window, BoxWindow)
        widths = np.diff(window.bounds.T, axis=0)
        p = self.p
        a = k - np.pi * p / widths
        b = k + np.pi * p / widths

        def compute_factor(x, k, widths):
            mask = x == 0
            if np.any(mask):
                out = np.zeros_like(k)
                widths_matrix = np.ones_like(k) * widths
                out[mask] = 0.5 * widths_matrix[mask]
                np.logical_not(mask, out=mask)
                out[mask] = np.sin(x[mask] * 0.5 * widths_matrix[mask]) / x[mask]
            else:
                out = np.sin(x * 0.5 * widths) / x
            return out

        res_a = compute_factor(a, k, widths)
        res_b = compute_factor(b, k, widths)

        res = (1j) ** (p + 1) * np.sqrt(2) * (res_a - (-1) ** p * res_b)
        ft = np.prod(res, axis=1)
        ft /= np.sqrt(window.volume)

        return ft


def multi_sinetaper_grid(d, p_component_max=2):
    r"""Given a class of taper `taper_p` of parameter `p` of :math:`\mathbb{R}^d`, return the list of taper `taper_p(p)` with :math:`p \in \{1, ..., P\}^d`.

    Args:
        d (int): Space dimension.

        taper_p (Class): Class of taper pf parameter p.

        p_component_max (int): Maximum component of the parameters :math:`p` of the family of tapers. Intuitively the number of taper used is :math:`P=\mathrm{p\_component\_max}^d`. Used only when ``tapers=None``. Defaults to 2.

    Returns:
        list: List of taper `taper_p(p)` with :math:`p \in \{1, ..., p_component_max\}^d`.
    """
    params = product(*(range(1, p_component_max + 1) for _ in range(d)))
    return [SineTaper(p) for p in params]
