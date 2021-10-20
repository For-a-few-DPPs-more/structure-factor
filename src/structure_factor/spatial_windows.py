#!/usr/bin/env python3
# coding=utf-8

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp
from rpy2 import robjects
from spatstat_interface.interface import SpatstatInterface

from structure_factor.utils import get_random_number_generator


class AbstractSpatialWindow(metaclass=ABCMeta):
    r"""Encapsulate the notion of spatial window in :math:`\mathbb{R}^d`.

    .. note::

        Typical usage:

        :py:class:`~structure_factor.point_pattern.PointPattern` has a :py:attr:`~structure_factor.point_pattern.PointPattern.window` argument/attribute.

    .. seealso::

        - :py:class:`~structure_factor.spatial_windows.BallWindow`, :py:class:`~structure_factor.spatial_windows.UnitBallWindow`
        - :py:class:`~structure_factor.spatial_windows.BoxWindow`, :py:class:`~structure_factor.spatial_windows.UnitBoxWindow`
    """

    @property
    @abstractmethod
    def dimension(self):
        """Return the ambient dimension of the corresponding window."""

    @property
    @abstractmethod
    def volume(self):
        """Compute the volume of the corresponding window."""

    @abstractmethod
    def __contains__(self, point):
        """Return True if :math:`d` dimensional vector ``point`` falls inside the corresponding :math:`d` dimensional window, otherwise return False.

        Args:
            point (numpy.ndarray): :math:`d` dimensional vector to test.
        """

    def indicator_function(self, points):
        r"""Return the indicator function of the corresponding window evaluated at each of the :math:`n` ``points``.

        Args:
            points (numpy.ndarray): vector of size :math:`d` or array of size :math:`n \times d` containing point(s) to be tested.
        Returns:
            bool or numpy.ndarray:
            - If :math:`n=1`, bool,
            - If :math:`n>1`, :math:`n` dimensional boolean array.
        """
        if points.ndim == 1 and points.size == self.dimension:
            return points in self
        return np.apply_along_axis(self.__contains__, axis=1, arr=points)

    @abstractmethod
    def rand(self, n=1, seed=None):
        r"""Generate `n` points uniformly at random in the corresponding spatial window.

        Args:
            n (int, optional): Number of points. Defaults to 1.
            seed (int or np.random.Generator, optional): Defaults to None.

        Returns:
            numpy.ndarray:
            - If :math:`n=1`, :math:`d` dimensional vector
            - If :math:`n>1`, :math:`n \times d` array containing the points.
        """


class BallWindow(AbstractSpatialWindow):
    """Create a :math:`d` dimensional ball window."""

    def __init__(self, center, radius=1.0):
        """Create a BallWindow.

        Args:
            center (numpy.ndarray): center of the ball.
            radius (float, optional): radius of the ball. Defaults to 1.0.
        """
        _center = np.array(center)
        if not _center.ndim == 1:
            raise ValueError("center must be 1D numpy.ndarray")
        if not radius > 0:
            raise ValueError("radius must be positive")
        self.center = _center
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

    def __contains__(self, point):
        assert point.ndim == 1 and point.size == self.dimension
        return self.indicator_function(point)

    def indicator_function(self, points):
        return np.linalg.norm(points - self.center, axis=-1) <= self.radius

    def rand(self, n=1, seed=None):
        rng = get_random_number_generator(seed)
        d = self.dimension
        if n == 1:
            points = rng.standard_normal(size=d + 2)
            points /= np.linalg.norm(points)
            return self.center + self.radius * points[:d]
        points = rng.standard_normal(size=(n, d + 2))
        points /= np.linalg.norm(points, axis=-1, keepdims=True)
        return self.center + self.radius * points[:, :d]

    def to_spatstat_owin(self, **params):
        """Convert the object to a ``spatstat.geom.disc`` R object of type ``disc``, which is a subtype of ``owin``.

        Args:
            params (dict): optional keyword arguments passed to ``spatstat.geom.disc``.

        Returns:
            spatstat.geom.disc: R object.

        .. seealso::

            `https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc <https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc>`_
        """
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        r = self.radius
        c = robjects.vectors.FloatVector(self.center)
        return spatstat.geom.disc(radius=r, centre=c, **params)


class UnitBallWindow(BallWindow):
    r"""Create a :math:`d` dimensional unit ball window.

    ``UnitBallWindow(center) = BallWindow(center, radius=1.0)``
    """

    def __init__(self, center):
        """Initialize unit ball window from ``center``.

        Args:
            center (numpy.ndarray): center of the ball.
        """
        super().__init__(center, radius=1.0)


class BoxWindow(AbstractSpatialWindow):
    r"""Create a :math:`d` dimensional box window :math:`\prod_{i=1}^{d} [a_i, b_i]` from ``bounds[i, :]`` :math:`=[a_i, b_i]`."""

    def __init__(self, bounds):
        r"""Initialize BoxWindow from ``bounds[i, :]`` :math:`=[a_i, b_i]`.

        Args:
            bounds (numpy.ndarray): :math:`d \times 2` array describing the bounds of the box row-wise.
        """
        _bounds = np.atleast_2d(bounds)
        if _bounds.ndim != 2:
            raise ValueError("bounds must be 2d numpy.ndarray")
        if _bounds.shape[1] != 2:
            raise ValueError("bounds must be d x 2 numpy.ndarray")
        if np.any(np.diff(_bounds, axis=-1) <= 0):
            raise ValueError("all bounds [a_i, b_i] must satisfy a_i < b_i")
        # use transpose to facilitate operations (unpacking, diff, rand, etc)
        self._bounds = np.transpose(_bounds)

    @property
    def bounds(self):
        r"""Return the bounds decribing the BoxWindow.

        ``bounds[i, :]`` :math:`=[a_i, b_i]`.
        """
        return np.transpose(self._bounds)

    @property
    def dimension(self):
        return self._bounds.shape[1]

    @property
    def volume(self):
        return np.prod(np.diff(self._bounds, axis=0))

    def __contains__(self, point):
        _point = np.array(point)
        assert _point.ndim == 1 and _point.size == self.dimension
        return self.indicator_function(_point)

    def indicator_function(self, points):
        a, b = self._bounds
        return np.logical_and(
            np.all(a <= points, axis=-1), np.all(points <= b, axis=-1)
        )

    def rand(self, n=1, seed=None):
        rng = get_random_number_generator(seed)
        a, b = self._bounds
        dim = self.dimension
        return rng.uniform(a, b, size=(dim,) if n == 1 else (n, dim))

    def to_spatstat_owin(self, **params):
        """Convert the object to a ``spatstat.geom.owin`` R object of type  ``owin``.

        Args:
            params (dict): optional keyword arguments passed to ``spatstat.geom.owin``.

        Returns:
            spatstat.geom.owin: R object.

        .. seealso::

            `https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/owin <https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/owin>`_
        """
        if self.dimension != 2:
            raise NotImplementedError("spatstat only handles 2D windows")
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        a, b = self._bounds
        x = robjects.vectors.FloatVector(a)
        y = robjects.vectors.FloatVector(b)
        return spatstat.geom.owin(xrange=x, yrange=y, **params)


class UnitBoxWindow(BoxWindow):
    r"""Create a :math:`d` dimensional unit box window :math:`\prod_{i=1}^{d} [c_i - \frac{1}{2}, c_i + \frac{1}{2}]` where :math:`c_i=` ``center[i]``.

    Default unit box is :math:`[0, 1]^d` (when ``center`` is None).
    """

    def __init__(self, d, center=None):
        r"""Create UnitBoxWindow, i.e., a BoxWindow with length equal to 1, in dimension ``d`` with center prescribed by ``center`` (defaults to :math:`[-\frac{1}{2}, \frac{1}{2}]^d`).

        Default window is :math:`[0, 1]^d`.

        Args:
            d (int): dimension of the box
            center (numpy.ndarray, optional): center of the box. Defaults to None.
        """
        _center = np.full(d, 0.5) if center is None else np.array(center)
        if _center.ndim != 1 or _center.size != d:
            raise ValueError("center must be 1D array with center.size == d")

        bounds = np.add.outer(_center, [-0.5, 0.5])
        super().__init__(bounds)
