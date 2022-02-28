"""The class :py:class:`~structure_factor.point_pattern.PointPattern` is designed to encapsulate one realization of a point process, along with the observation window, and the intensity of the point process.

- :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`: Restrict the point pattern to a specific window.
- :py:meth:`~structure_factor.point_pattern.PointPattern.convert_to_spatstat_ppp`: Converts the point pattern into a ``spatstat.geom.ppp`` R object.
- :py:meth:`~structure_factor.point_pattern.PointPattern.plot`: Plots the point pattern.

.. note::

    **Typical usage**

    - The class :py:class:`~structure_factor.structure_factor.StructureFactor` gets initialized using a :py:class:`~structure_factor.point_pattern.PointPattern`.
"""

import matplotlib.pyplot as plt
import numpy as np
from rpy2 import robjects
from spatstat_interface.interface import SpatstatInterface

from structure_factor.spatial_windows import AbstractSpatialWindow


class PointPattern(object):
    r"""Encapsulate one realization of a point process, the corresponding observation window, and the intensity of the underlying point process.

    .. todo::

        list attributes

    Example:
        .. plot:: code/point_pattern/point_pattern.py
            :include-source: True
            :align: center

    .. seealso::

        - :py:mod:`~structure_factor.spatial_windows`
        - :py:mod:`~structure_factor.point_processes`
        - :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`
        - :py:meth:`~structure_factor.point_pattern.PointPattern.plot`
    """

    #! what are **params?
    def __init__(self, points, window, intensity=None, **params):
        r"""Initialize the object from a realization ``points`` of the underlying point process with intensity ``intensity`` observed in ``window``.

        Args:
            points (numpy.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d` consisting of a realization of a point process.

            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): Observation window containing the ``points``.

            intensity(float, optional): Intensity of the point process. If None, the intensity of the point process is approximated by the ratio of the number of point to the window volume. Defaults to None.

        Keyword Args:
            params: Possible additional parameters of the point process.
        """
        points = np.asarray(points)
        assert points.ndim == 2
        self.points = points

        assert isinstance(window, AbstractSpatialWindow)
        self.window = window

        if intensity is None:
            intensity = self.points.shape[0] / window.volume
        assert intensity > 0
        self.intensity = float(intensity)

        self.params = params

    @property
    def dimension(self):
        """Ambient dimension of the space where the points live."""
        return self.points.shape[1]

    def restrict_to_window(self, window):
        """Return a new instance of :py:class:`~structure_factor.point_pattern.PointPattern` with the following attributes,

        - points: points of the original object that fall inside the prescribed ``window``,
        - window: prescribed ``window``,
        - intensity: intensity of the original object.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): New observation window to restrict to.

        Returns:
            :py:class:`~structure_factor.point_pattern.PointPattern`: Restriction of the initial ``PointPattern`` instance to the prescribed ``window``.

        Example:
            .. plot:: code/point_pattern/restrict_pp.py
                :include-source: True
                :align: center

        .. seealso::

            - :py:mod:`~structure_factor.spatial_windows`
            - :py:mod:`~structure_factor.point_processes`
            - :py:meth:`~structure_factor.point_pattern.PointPattern.plot`
        """
        assert isinstance(window, AbstractSpatialWindow)
        mask = window.indicator_function(self.points)
        return PointPattern(self.points[mask], window, self.intensity)

    def convert_to_spatstat_ppp(self, **params):
        """Convert the object into a point pattern ``spatstat.geom.ppp`` R object.

        Keyword Args:
            params (dict): Optional keyword arguments passed to ``spatstat.geom.ppp``.

        Returns:
            Point pattern R object of type ``spatstat.geom.ppp``.

        .. seealso::

            - `https://rdrr.io/cran/spatstat.geom/man/ppp.html <https://rdrr.io/cran/spatstat.geom/man/ppp.html>`_
        """
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        window = params.get("window", self.window)
        if window is not None and isinstance(window, AbstractSpatialWindow):
            params["window"] = window.to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)

    def plot(self, axis=None, window=None, show_window=False, file_name="", **kwargs):
        """Scatter plot of :py:attr:`~structure_factor.point_pattern.PointPattern.points`.

        Args:
            axis (plt.Axes, optional): Support axis of the plot. Defaults to None.

            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): Output observation window. Defaults to None.

            show_window (bool, optional): Display the ``window``, ambient dimension should be 2.

        Returns:
            plt.Axes: Plot axis.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        if window is None:
            window = self.window
            points = self.points
        else:
            assert isinstance(window, AbstractSpatialWindow)
            mask = window.indicator_function(self.points)
            points = self.points[mask]

        if show_window:
            window.plot(axis=axis)

        kwargs.setdefault("c", "k")
        kwargs.setdefault("s", 0.5)
        axis.scatter(points[:, 0], points[:, 1], **kwargs)
        axis.set_aspect("equal")

        if file_name:
            fig = axis.get_figure()
            fig.savefig(file_name, bbox_inches="tight")
        return axis
