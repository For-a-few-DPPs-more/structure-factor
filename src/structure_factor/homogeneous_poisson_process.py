#!/usr/bin/env python3
# coding=utf-8

from structure_factor.spatial_windows import AbstractSpatialWindow, UnitBoxWindow
from structure_factor.utils import get_random_number_generator


class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson Point Process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_"""

    def __init__(self, intensity=1.0):
        """Create a `homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_ with prescribed (positive) ``intensity`` parameter.

        :param intensity: Constant intensity parameter of the homogeneous Poisson point process, defaults to 1.0
        :type intensity: real, optional
        """
        if not intensity > 0:
            raise TypeError("intensity argument must be positive")
        self.intensity = intensity

    def generate_sample(self, window=UnitBoxWindow(2), random_state=None):
        """Generate an exact sample from the corresponding :py:class:`HomogeneousPoissonPointProcess` restricted to `window`."""
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(random_state)
        nb_points = rng.poisson(self.intensity * window.volume)
        return window.rand(nb_points, random_state=rng)
