from structure_factor.spatial_windows import (
    AbstractSpatialWindow,
    BallWindow,
    BoxWindow,
)
from structure_factor.utils import get_random_number_generator
import numpy as np


class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_."""

    def __init__(self, intensity=1.0):
        """Create a `homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_ with prescribed (positive) ``intensity`` parameter.

        :param intensity: Constant intensity parameter of the homogeneous Poisson point process, defaults to 1.0
        :type intensity: real, optional
        """
        if not intensity > 0:
            raise TypeError("intensity argument must be positive")
        self.intensity = intensity

    def structure_factor(self, k):
        """Structure factor of the Poisson point process

        Args:
            k (np.array): Points to evaluate on.

        Returns:
            np.array: Structure factor of Poisson process evaluated on `k`.
        """
        return np.ones_like(k)

    def pair_correlation_function(self, r):
        """Pair correlation function of the Poisson point process

        Args:
            r (np.array): Points to evaluate on.

        Returns:
            np.array: Structure factor of Poisson process evaluated on `k`.
        """
        return np.ones_like(r)

    def generate_sample(self, window, seed=None):
        r"""Generate an exact sample from the corresponding :py:class:`~structure_factor.homogeneous_poisson_process.HomogeneousPoissonPointProcess` restricted to the :math:`d` dimensional `window`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): Observation window where the points will be generated.

        Returns:
            numpy.ndarray: of size :math:`n \times d`, where :math:`n` is the number of points forming the sample.

        .. plot:: code/poisson_sample_example.py
            :include-source: True
            :caption:
            :alt: alternate text
            :align: center
        """
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)
        nb_points = rng.poisson(self.intensity * window.volume)
        return window.rand(nb_points, seed=rng)

        # todo add structure factor and pair correlation function

    # todo move all structure factor and pair correlation functions here for ginibre add note for the simulation to use DPPY