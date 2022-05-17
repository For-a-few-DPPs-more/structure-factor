"""Collection of classes representing observation windows (box, ball, etc).

- :py:class:`~structure_factor.spatial_windows.BallWindow`: Ball window object.
- :py:class:`~structure_factor.spatial_windows.BoxWindow`: Box window object.

.. note::

    **Typical usage**

    - :py:class:`~structure_factor.point_pattern.PointPattern` has a :py:attr:`~structure_factor.point_pattern.PointPattern.window` argument/attribute.
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from rpy2 import robjects
from spatstat_interface.interface import SpatstatInterface

from structure_factor.utils import get_random_number_generator


class AbstractSpatialWindow(metaclass=ABCMeta):
    r"""Encapsulate the notion of spatial window in :math:`\mathbb{R}^d`."""

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
        r"""Return True if :math:`d` dimensional vector ``point`` falls inside the corresponding :math:`d` dimensional window, otherwise return False.

        Args:
            point (numpy.ndarray): :math:`d` dimensional vector to test.
        """

    def indicator_function(self, points):
        r"""Return the indicator function of the corresponding window evaluated at each of the :math:`n` ``points``.

        Args:
            points (numpy.ndarray): Vector of size :math:`d` or array of size :math:`n \times d` containing the point(s) to be tested.

        Returns:
            bool or numpy.ndarray:
            - If :math:`n=1`, bool.
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
            - If :math:`n=1`, :math:`d` dimensional vector.
            - If :math:`n>1`, :math:`n \times d` array containing the points.
        """


class BallWindow(AbstractSpatialWindow):
    r"""Create a :math:`d` dimensional ball window :math:`B(c, r)`, where :math:`c \in \mathbb{R}^d` and :math:`r>0`.

    .. todo::

        list attributes

    Example:
        .. plot:: code/spatial_window/ball_window.py
            :include-source: True
            :align: center

    .. seealso::

        - :py:mod:`~structure_factor.point_pattern`
        - :py:class:`~structure_factor.spatial_windows.BoxWindow`
    """

    def __init__(self, center, radius=1.0):
        """Initialize a :math:`d` dimensional ball window :math:`B(c, r)` from the prescribed ``center`` and ``radius``.

        Args:
            center (numpy.ndarray): Center :math:`c` of the ball.
            radius (float, optional): Radius :math:`r > 0` of the ball. Defaults to 1.0.
        """
        center = np.asarray(center)
        if not center.ndim == 1:
            raise ValueError("center must be 1D numpy.ndarray")
        if not radius > 0:
            raise ValueError("radius must be positive")
        self.center = center
        self.radius = float(radius)

    @property
    def dimension(self):
        return len(self.center)

    @property
    def surface(self):
        d, r = self.dimension, self.radius
        if d == 1:
            return 0.0
        if d == 2:
            return 2 * np.pi * r
        if d == 3:
            return 4 * np.pi * r ** 2
        return 2 * np.pi ** (d / 2) * r ** (d - 1) / sp.special.gamma(d / 2)

    @property
    def volume(self):
        d, r = self.dimension, self.radius
        if d == 1:
            return 2 * r
        if d == 2:
            return np.pi * r ** 2
        if d == 3:
            return 4 / 3 * np.pi * r ** 3
        return np.pi ** (d / 2) * r ** d / sp.special.gamma(d / 2 + 1)

    def __contains__(self, point):
        point = np.asarray(point)
        assert point.ndim == 1 and point.size == self.dimension
        return self.indicator_function(point)

    def indicator_function(self, points):
        return np.linalg.norm(points - self.center, axis=-1) <= self.radius

    def rand(self, n=1, seed=None):
        # Method of dropped coordinates
        # Efficiently sampling vectors and coordinates from the n-sphere and n-ball
        # Voelker, Aaron and Gosmann, Jan and Stewart, Terrence
        # doi: 10.13140/RG.2.2.15829.01767/1
        rng = get_random_number_generator(seed)
        d = self.dimension
        points = rng.standard_normal(size=(n, d + 2))
        points /= np.linalg.norm(points, axis=-1, keepdims=True)
        idx = 0 if n == 1 else slice(0, n)
        return self.center + self.radius * points[idx, :d]

    def to_spatstat_owin(self, **params):
        """Convert the object to a ``spatstat.geom.disc`` R object of type ``disc``, which is a subtype of ``owin``.

        Args:
            params (dict): Optional keyword arguments passed to ``spatstat.geom.disc``.

        Returns:
            spatstat.geom.disc: R object.

        .. seealso::

            - `https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc <https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/disc>`_
        """
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        r = self.radius
        c = robjects.vectors.FloatVector(self.center)
        return spatstat.geom.disc(radius=r, centre=c, **params)

    def plot(self, axis=None, **kwargs):
        """Display the window on matplotlib `axis`.

        Args:
            axis (plt.Axes, optional): Support axis of the plot. Defaults to None.

        Keyword Args:
            kwargs (dict): Keyword arguments of ``matplotlib.patches.Circle`` with default ``fill=False``.

        Returns:
            plt.Axes: Plot axis.
        """
        if self.dimension != 2:
            raise NotImplementedError("Method implemented only for 2D window")

        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        kwargs.setdefault("fill", False)
        circle = Circle(self.center, self.radius, **kwargs)

        axis.add_patch(circle)
        return axis


class UnitBallWindow(BallWindow):
    r"""Create a d-dimensional unit ball window :math:`B(c, r=1)`, where :math:`c \in \mathbb{R}^d`.

    .. note::

        ``UnitBallWindow(center) = BallWindow(center, radius=1.0)``
    """

    def __init__(self, center):
        """Initialize a :math:`d` dimensional unit ball window :math:`B(c, r=1)` from the prescribed ``center``.

        Args:
            center (numpy.ndarray, optional): Center :math:`c` of the ball.
        """
        super().__init__(center, radius=1.0)


class BoxWindow(AbstractSpatialWindow):
    r"""Create a :math:`d` dimensional box window :math:`\prod_{i=1}^{d} [a_i, b_i]`.

    .. todo::

        list attributes

    Example:
        .. plot:: code/spatial_window/box_window.py
            :include-source: True
            :align: center

    .. seealso::

        - :py:mod:`~structure_factor.point_pattern`
        - :py:class:`~structure_factor.spatial_windows.BoxWindow`
    """

    def __init__(self, bounds):
        r"""Initialize :math:`d` dimensional unit box window the prescibed  ``bounds[i, :]`` :math:`=[a_i, b_i]`.

        Args:
            bounds (numpy.ndarray): :math:`d \times 2` array describing the bounds of the box.
        """
        bounds = np.atleast_2d(bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must be d x 2 numpy.ndarray")
        if np.any(np.diff(bounds, axis=-1) <= 0):
            raise ValueError("all bounds [a_i, b_i] must satisfy a_i < b_i")
        # use transpose to facilitate operations (unpacking, diff, rand, etc)
        self._bounds = np.transpose(bounds)

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
        point = np.asarray(point)
        assert point.ndim == 1 and point.size == self.dimension
        return self.indicator_function(point)

    def indicator_function(self, points):
        a, b = self._bounds
        return np.logical_and(
            np.all(a <= points, axis=-1), np.all(points <= b, axis=-1)
        )

    def rand(self, n=1, seed=None):
        rng = get_random_number_generator(seed)
        a, b = self._bounds
        d = self.dimension
        return rng.uniform(a, b, size=(d,) if n == 1 else (n, d))

    def to_spatstat_owin(self, **params):
        """Convert the object to a ``spatstat.geom.owin`` R object of type  ``owin``.

        Args:
            params (dict): Optional keyword arguments passed to ``spatstat.geom.owin``.

        Returns:
            spatstat.geom.owin: R object.

        .. seealso::

            - `https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/owin <https://rdocumentation.org/packages/spatstat.geom/versions/2.2-0/topics/owin>`_
        """
        if self.dimension != 2:
            raise NotImplementedError("spatstat only handles 2D windows")
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        a, b = self.bounds
        x = robjects.vectors.FloatVector(a)
        y = robjects.vectors.FloatVector(b)
        return spatstat.geom.owin(xrange=x, yrange=y, **params)

    def plot(self, axis=None, **kwargs):
        """Display the window on matplotlib `axis`.

        Args:
            axis (plt.Axes, optional): Support axis of the plot. Defaults to None.

        Keyword Args:
            kwargs (dict): Keyword arguments of ``matplotlib.patches.Rectangle`` with default ``fill=False``.

        Returns:
            plt.Axes: Plot axis.
        """
        if self.dimension != 2:
            raise NotImplementedError("Method implemented only for 2D window")

        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        kwargs.setdefault("fill", False)

        xy = self._bounds[0]
        width, height = np.diff(self._bounds, axis=0).ravel()
        rectangle = Rectangle(xy, width, height, **kwargs)

        axis.add_patch(rectangle)
        return axis


class UnitBoxWindow(BoxWindow):
    r"""Create a :math:`d` dimensional unit box window :math:`\prod_{i=1}^{d} [c_i - \frac{1}{2}, c_i + \frac{1}{2}]` where :math:`c \in \mathbb{R}^d`."""

    def __init__(self, center):
        r"""Initialize a :math:`d` dimensional unit box window :math:`\prod_{i=1}^{d} [c_i - \frac{1}{2}, c_i + \frac{1}{2}]`, i.e., a box window with length equal to 1 and prescribed ``center``, such that :math:`c_i=` ``center[i]``.

        Args:
            center (numpy.ndarray): Center :math:`c` of the box.
        """
        if np.ndim(center) != 1:
            raise ValueError("center must be 1D array.")

        bounds = np.add.outer(center, [-0.5, 0.5])
        super().__init__(bounds)


def check_cubic_window(window):
    """Check whether ``window`` is cubic.

    Args:
        window (:py:class:`~structure_factor.spatial_windows.BoxWindow`): Window to be checked.

    Example:
        .. literalinclude:: code/spatial_window/check_cubic_window.py
            :language: python

    """
    if not isinstance(window, BoxWindow):
        raise TypeError("window must be an instance of BoxWindow.")
    lengths = np.diff(window.bounds, axis=1)
    if np.any(lengths != lengths[0]):
        raise ValueError("window should be a 'cubic' BoxWindow.")
    return None


def check_centered_window(window):
    """Check whether ``window`` is centered at the origin.

    Args:
        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Window to be checked.

    Example:
        .. literalinclude:: code/spatial_window/check_centered_window.py
            :language: python

    """
    if isinstance(window, BoxWindow):
        if np.sum(window.bounds) != 0:
            raise ValueError(
                "window is not centered at the origin. Hint: use a centered window."
            )
    if isinstance(window, BallWindow):
        if any(window.center != 0):
            raise ValueError(
                "window is not centered at the origin. Hint: use a centered window."
            )
    return None


def subwindow_parameter_max(window, subwindow_type="BoxWindow"):
    """
    Return the parameter i.e., lengthside (resp. radius) of the largest cubic (resp. ball)  subwindow of ``window``.

    Args:
        window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): BoxWindow or BallWindow centered at the origin.
        subwindow_type (str, optional): Type of the subwindow ("BoxWindow" or "BallWindow"). Defaults to "BoxWindow".


    Returns:
        float : Parameter of the largest subwindow of ``window``.

    Example:
        .. plot:: code/spatial_window/subwindow_param_max.py
            :include-source: True
            :align: center
    """
    if subwindow_type not in ["BoxWindow", "BallWindow"]:
        raise ValueError(
            "The available subwindow types are BallWindow or BoxWindow. Hint: the parameter corresponding to the window type must be 'BallWindow' or 'BoxWindow'. "
        )
    check_centered_window(window)
    # window parameter
    if isinstance(window, BallWindow):
        if subwindow_type == "BallWindow":
            param_max = window.radius
        else:
            param_max = window.radius * 2 / np.sqrt(2)
            # length side of the BoxWindow
    elif isinstance(window, BoxWindow):
        if subwindow_type == "BallWindow":
            param_max = np.min(np.diff(window.bounds)) / 2
        else:
            param_max = np.min(np.diff(window.bounds))
    return param_max
