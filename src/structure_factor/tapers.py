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
    # print(x.shape)
    # print(teta.shape)
    # teta = p * (x / l * 0.5 + np.pi * 0.5)
    sin_teta = np.sin(teta)  # shape n*d
    taper_p = window.indicator_function(x).astype(float) * np.prod(
        sin_teta, axis=1
    )  # shape n*1
    # taper_p /= np.sqrt(window.volume)
    return taper_p