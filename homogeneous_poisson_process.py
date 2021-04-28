#!/usr/bin/env python3
# coding=utf-8

from spatial_windows import AbstractSpatialWindow
from utils import get_random_number_generator


class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson Point Process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_
    """
    def __init__(self, intensity=1.0):
        if not intensity > 0:
            raise TypeError("intensity argument must be positive")
        self.intensity = intensity

    def generate_sample(self, window, random_state=None):
        """Generate an exact sample from the corresponding :py:class:`HomogeneousPoissonPointProcess` restricted to `window` (default is `HomogeneousPoissonPointProcess.window`).
        """
        rng = get_random_number_generator(random_state)
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")
        nb_points = rng.poisson(self.intensity * window.volume)
        return window.rand(nb_points, random_state=rng)
