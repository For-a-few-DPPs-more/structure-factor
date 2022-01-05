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

    def ft_taper_1(self, k, window):
        widths = np.diff(window.bounds.T, axis=0)
        print(widths.shape)
        p = self.p
        A = 1 / np.sqrt(window.volume)
        A_2 = np.sqrt(2) * 0.5
        a = k - np.pi * p / widths
        b = k + np.pi * p / widths
        a1 = 1j * widths * 0.5 * k
        b2 = 1j * np.pi * p
        res_1 = (np.exp(b2 - a1) - np.exp(a1)) / a
        res_2 = (np.exp(a1) - np.exp(-b2 - a1)) / b
        res = (res_1 - res_2) * A_2
        ft = np.prod(res, axis=1)
        ft = ft * A
        print(np.max(np.abs(ft)))
        return ft

    def ft_taper(self, k, window):
        widths = np.diff(window.bounds.T, axis=0)  # l
        p = self.p
        print("p", p)
        a = k - np.pi * p / widths  # k- pi*p/l
        b = k + np.pi * p / widths  # k + pi*p/l
        res_1 = (
            (-1j) ** p * np.sin(b * 0.5 * widths) / b
        )  # (-i)^p sin(( k + np.pi * p / widths)* (l/2)) / (k + pi*p/l)
        res_2 = (
            (1j) ** p * np.sin(a * 0.5 * widths) / a
        )  # i^p sin(( k - np.pi * p / widths)* (l/2)) / (k - pi*p/l)
        # print(np.max(res_1), np.max(res_2))
        res = (2 * np.sqrt(2) * np.pi * 1j) * (res_1 - res_2)
        print("k", k[11320:11330, :])
        ft = np.prod(res, axis=1)
        print("ft", ft[11320:11330])
        ft = ft / np.sqrt(window.volume)
        print("norm", ft[11320:11330])
        return ft

    def ft_taper_old_version(self, k, window):
        # todo old version to delete
        widths = np.diff(window.bounds.T, axis=0)
        p = self.p
        a = k * widths - np.pi * p
        b = k * widths + np.pi * p
        # print(np.min(abs(a)), np.min(abs(b)))
        c = -2 * np.pi * 1j
        sines = (np.sin(a * 0.5) / a - (-1) ** p * np.sin(b * 0.5) / b) * c
        sines *= np.exp(1j * np.pi * p * 0.5)
        ft = np.prod(sines, axis=1)
        ft = ft * np.sqrt(window.volume)
        return ft


def sin_taper(p, x, window):
    """sin taper family
    Args:
        p (nd.array): 1*d array
        x ([type]): n*d array
        window ([type]): [description]
    Returns:
        [type]: 1*d array
    """
    l = np.diff(window.bounds.T, axis=0)  # shape 1*d
    teta = np.pi * p * (x / l + 0.5)  # shape n*d
    # teta = p * (x / l * 0.5 + np.pi * 0.5)
    sin_teta = np.sin(teta)  # shape n*d
    taper_p = window.indicator_function(x).astype(float) * np.prod(
        sin_teta, axis=1
    )  # shape n*1
    taper_p /= np.sqrt(window.volume)
    return taper_p
