r"""Collection of classes representing tapers (also called tapering functions or window functions) with two methods:

- ``.taper(x, window)`` corresponding to the tapering function :math:`t(x, W)`,

- ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the tapering function.

These tapers satisfy some conditions listed in :cite:`DGRR:22` (Sections 3.1, 4.3).

Any taper to be used should be a class with the two methods ``.taper(x, window)`` and ``.ft_taper(k, window)``

Brief example :

     .. literalinclude:: code/structure_factor/taper.py
        :language: python
"""

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
            k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the Fourier transform :math:`t(k, W)` is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Return:
            numpy.ndarray: Evaluation of the Fourier transform :math:`t(k, W)`.
        """
        assert isinstance(window, BoxWindow)
        widths = 0.5 * np.diff(window.bounds.T, axis=0)
        if (k == 0).any():
            sines = np.zeros_like(k)
            widths_matrix = np.ones_like(k) * widths
            sines[k == 0] = 2.0 * widths_matrix[k == 0]
            sines[k != 0] = 2.0 * np.sin(k[k != 0] * widths_matrix[k != 0]) / k[k != 0]
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
            k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the Fourier transform :math:`t(k, W)` is evaluated.

            window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): :math:`d`-dimensional rectangular window :math:`W`.

        Return:
            numpy.ndarray: Evaluation of the Fourier transform :math:`t(k, W)`.
        """
        assert isinstance(window, BoxWindow)
        widths = np.diff(window.bounds.T, axis=0)
        p = self.p
        a = k - np.pi * p / widths
        b = k + np.pi * p / widths
        if (a == 0).any():
            res_1 = np.zeros_like(k)
            widths_matrix = np.ones_like(k) * widths
            res_1[a == 0] = 0.5 * widths_matrix[a == 0]
            res_1[a != 0] = np.sin(a[a != 0] * 0.5 * widths_matrix[a != 0]) / a[a != 0]
        else:
            res_1 = np.sin(a * 0.5 * widths) / a
        if (b == 0).any():
            res_2 = np.zeros_like(k)
            widths_matrix = np.ones_like(k) * widths
            res_2[b == 0] = 0.5 * widths_matrix[b == 0]
            res_2[b != 0] = np.sin(b[b != 0] * 0.5 * widths_matrix[b != 0]) / b[b != 0]
        else:
            res_2 = np.sin(b * 0.5 * widths) / b
        res = (1j) ** (p + 1) * np.sqrt(2) * (res_1 - (-1) ** p * res_2)
        ft = np.prod(res, axis=1)
        ft /= np.sqrt(window.volume)
        return ft
