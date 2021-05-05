#!/usr/bin/env python3
# coding=utf-8

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp

from utils import get_random_number_generator


class AbstractSpatialWindow(metaclass=ABCMeta):

    @property
    @abstractmethod
    def dimension(self):
        """Return the dimension of the corresponding :py:class:`AbstractSpatialWindow`
        """

    @property
    @abstractmethod
    def volume(self):
        """Compute the volume of the corresponding :py:class:`AbstractSpatialWindow`
        """

    @abstractmethod
    def indicator_function(self, points):
        """Return a boolean or 1D np.array indicating which points lie in the corresponding :py:class:`AbstractSpatialWindow.
        """

    @abstractmethod
    def rand(self, nb_points=1, random_state=None):
        """Generate `nb_points` points uniformly at random in the corresponding :py:class:`AbstractSpatialWindow`
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
            return np.pi * r**2
        return np.pi**(d/2) * r**d / sp.special.gamma(d/2 + 1)

    def indicator_function(self, points):
        return np.linalg.norm(points - self.center, axis=-1) <= self.radius

    def rand(self, nb_points=1, random_state=None):
        rng = get_random_number_generator(random_state)
        d = self.dimension
        if nb_points == 1:
            points = rng.standard_normal(size=d+2)
            points /= np.linalg.norm(points)
            return self.center + self.radius * points[:d]
        points = rng.standard_normal(size=(nb_points, d+2))
        points /= np.linalg.norm(points, axis=1)[:, None]
        return self.center + self.radius * points[:, :d]


class UnitBallWindow(BallWindow):
    def __init__(self, center):
        super().__init__(center, radius=1.0)


class BoxWindow(AbstractSpatialWindow):
    def __init__(self, bounds):
        if bounds.ndim != 2:
            raise ValueError("bounds must be 2D np.ndarray")
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
        return np.logical_and(np.all(points >= self.bounds[0], axis=-1),
                              np.all(points <= self.bounds[1], axis=-1))

    def rand(self, nb_points=1, random_state=None):
        rng = get_random_number_generator(random_state)
        if nb_points == 1:
            return rng.uniform(*self.bounds)
        return rng.uniform(*self.bounds, size=(nb_points, self.dimension))


class UnitBoxWindow(BoxWindow):
    def __init__(self, dimension, center=None):
        if center is None:
            center = np.full(dimension, 0.5)
        elif dimension != len(center):
            raise ValueError("Dimension mismatch: dimension != len(center)")

        bounds = np.array([[-0.5], [0.5]]) + center
        super().__init__(bounds)
