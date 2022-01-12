import numpy as np

from structure_factor.spatial_windows import BoxWindow

# class AbstractTaper:
#     @staticmethod
#     def taper(x, window):
#         r"""Evaluate the

#         Args:
#             x (np.ndarray): Point array (d,) or array of points (n, d).
#             window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window :math:`W`.

#         Returns:
#             array_like: evaluation of the taper :math:`h(x, W)`
#         """
#         raise NotImplementedError()

#     @staticmethod
#     def ft_taper(k, window):
#         r"""Evaluate the Fourier transform of the taper :math:`h` :py:meth:`~structure_factor/tapers.Abstract.taper`.

#         .. math::

#             H(k) = F[h](k)
#             = \int_{\mathbb{R}^d}
#                 h(x) \exp(-i \left\langle k, x \right\rangle) d x.

#         Args:
#             k (np.ndarray): Wave vector array (d,) or array of wave vectors (n, d).

#             window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.

#         Return:
#             numpy.array: Evaluation of the Fourier transform :math:`H(k)`.
#         """
#         raise NotImplementedError()


class BartlettTaper:
    @staticmethod
    def taper(x, window):
        r"""Evaluate the indicator function of a rectangular ``window`` :math:`W` on points ``x`` divided by the square root of the volume :math:`|W|`.
        .. math::
            h(x, W) = \frac{1}{\sqrt{|W|}} 1_{x \in W}.
        Args:
            x (np.ndarray): Points.
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window :math:`W`.
        Returns:
            array_like: evaluation of the taper :math:`h(x, W)`
        """
        assert isinstance(window, BoxWindow)
        h = window.indicator_function(x).astype(float)
        h /= np.sqrt(window.volume)
        return h

    @staticmethod
    def ft_taper(k, window):
        r"""Evaluate the Fourier transform of :py:meth:`~structure_factor/tapers.BartlettTaper.taper`.
        .. math::
            H(k) = F[h](k)
            = \int_{\mathbb{R}^d}
                h(x) \exp(-i \left\langle k, x \right\rangle) d x.
        Args:
            k (np.ndarray): Wave vector array (d,) or array of wave vectors (n, d).
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window.
        Return:
            numpy.array: Evaluation of the Fourier transform :math:`H(k)`.
        """
        widths = 0.5 * np.diff(window.bounds.T, axis=0)
        sines = 2.0 * np.sin(k * widths) / k
        ft = np.prod(sines, axis=1)
        ft /= np.sqrt(window.volume)
        return ft


class SineTaper:
    def __init__(self, p):
        self.p = np.array(p)

    def taper(self, x, window):
        widths = np.diff(window.bounds.T, axis=0)
        # d = x.shape[1]
        sines = x / widths + 0.5
        sines *= np.pi * self.p
        np.sin(sines, out=sines)

        h_x = 2 * window.indicator_function(x).astype(float)
        h_x *= np.prod(sines, axis=1)
        h_x *= np.sqrt(1 / window.volume)  # normalization
        return h_x

    def ft_taper(self, k, window):
        widths = np.diff(window.bounds.T, axis=0)  # l
        p = self.p
        a = k - np.pi * p / widths  # k- pi*p/l
        b = k + np.pi * p / widths  # k + pi*p/l

        res_1 = np.sin(a * 0.5 * widths) / a
        #  sin(( k - np.pi * p / widths)* (l/2)) / (k - pi*p/l)
        res_2 = np.sin(b * 0.5 * widths) / b
        # sin(( k + np.pi * p / widths)* (l/2)) / (k + pi*p/l)
        res = (
            (1j) ** (p + 1) * np.sqrt(2) * (res_1 - (-1) ** p * res_2)
        )  # i^(p+1) sqrt(2) (res_1 - (-1)^p res_2)
        ft = np.prod(res, axis=1)
        ft = ft / np.sqrt(window.volume)
        return ft
