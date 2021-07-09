#!/usr/bin/env python3
# coding=utf-8

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp
from rpy2 import robjects

from structure_factor.utils import get_random_number_generator
from structure_factor.spatstat_interface import SpatstatInterface


class AbstractSpatialWindow(metaclass=ABCMeta):
    @property
    @abstractmethod
    def dimension(self):
        """Return the ambient dimension of the corresponding window."""

    @property
    @abstractmethod
    def volume(self):
        """Compute the volume of the corresponding window."""

    @abstractmethod
    def indicator_function(self, points):
        """Indicator function returning a boolean or boolean array indicating which points lie in the corresponding :py:class:`AbstractSpatialWindow.

        Args:
            points (np.ndarray): Points to be tested.
        """

    @abstractmethod
    def rand(self, n=1, random_state=None):
        r"""Generate `n` points uniformly at random in the corresponding spatial window

        Args:
            n (int, optional): Number of points. Defaults to 1.
            random_state (optional): Defaults to None.

        Returns:
            points (np.ndarray):
            If :math:`n=1`, :math:`d` dimensional vector
            If :math:`n>1`, :math:`n \times d` array containing the points
        """


class BallWindow(AbstractSpatialWindow):
    def __init__(self, center, radius=1.0):
        if not center.ndim == 1:
            raise ValueError("center must be 1D np.ndarray")
        if not radius > 0:
            raise ValueError("radius must be positive")
        self.center = center
        self.radius = radius

    @property
    def dimension(self):
        return len(self.center)

    @property
    def volume(self):
        d, r = self.dimension, self.radius
        if d == 1:
            return 2 * r
        if d == 2:
            return np.pi * r ** 2
        return np.pi ** (d / 2) * r ** d / sp.special.gamma(d / 2 + 1)

    def indicator_function(self, points):
        return np.linalg.norm(points - self.center, axis=-1) <= self.radius

    def rand(self, n=1, random_state=None):
        rng = get_random_number_generator(random_state)
        d = self.dimension
        if n == 1:
            points = rng.standard_normal(size=d + 2)
            points /= np.linalg.norm(points)
            return self.center + self.radius * points[:d]
        points = rng.standard_normal(size=(n, d + 2))
        points /= np.linalg.norm(points, axis=1)[:, None]
        return self.center + self.radius * points[:, :d]

    def convert_to_spatstat_owin(self, **params):
        # https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        r = self.radius
        c = robjects.vectors.FloatVector(self.center)
        return spatstat.geom.disc(radius=r, centre=c, **params)


class UnitBallWindow(BallWindow):
    def __init__(self, center):
        super().__init__(center, radius=1.0)


class BoxWindow(AbstractSpatialWindow):
    def __init__(self, bounds):
        """Create a :math:`d` dimensional box window :math:`\prod_{i=1}^{d} [a_i, b_i]` from ``bounds[:, i]`` :math:`=[a_i, b_i]`.

        :param bounds: 2 x d array describing the bounds of the box columnwise
        :type bounds: np.ndarray
        """
        if bounds.ndim != 2:
            raise ValueError("bounds must be 2d np.ndarray")
        if bounds.shape[0] != 2:
            raise ValueError("bounds must be 2xd np.ndarray")
        self.bounds = bounds

    @property
    def dimension(self):
        return self.bounds.shape[1]

    @property
    def volume(self):
        return np.prod(np.diff(self.bounds, axis=0))

    def indicator_function(self, points):
        return np.logical_and(
            np.all(points >= self.bounds[0], axis=-1),
            np.all(points <= self.bounds[1], axis=-1),
        )

    def rand(self, n=1, random_state=None):
        rng = get_random_number_generator(random_state)
        if n == 1:
            return rng.uniform(*self.bounds)
        return rng.uniform(*self.bounds, size=(n, self.dimension))

    def convert_to_spatstat_owin(self, **params):
        # https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/owin
        assert self.dimension == 2
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.bounds[:, 0])
        y = robjects.vectors.FloatVector(self.bounds[:, 1])
        return spatstat.geom.owin(xrange=x, yrange=y, **params)


class UnitBoxWindow(BoxWindow):
    def __init__(self, d, center=None):
        """Create a ``d`` dimensional unit box window :math:`\prod_{i=1}^{d} [c_i - \\frac{1}{2}, c_i + \\frac{1}{2}]` where :math:`c_i=` ``center[i]``.
        Default unit box is :math:`[0, 1]^d` (when ``center=None``).

        :param d: dimension of the box
        :type d: int
        :param center: center of the box, defaults to None.
        :type center: np.ndarray, optional
        """
        if center is None:
            center = np.full(d, 0.5)
        elif center.ndim != 1 or center.size != d:
            raise ValueError("Dimension mismatch: center.size != d")

        bounds = np.array([[-0.5], [0.5]]) + center
        super().__init__(bounds)
