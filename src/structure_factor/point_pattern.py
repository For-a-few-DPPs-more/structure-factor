"""
Object designed to encapsulate a realization from a point process, the observation window, and the intensity of the point process. We denote the final encapsulation by a point pattern.

**This class contains also**:
    - :py:meth:`restrict_to_window`: Restrict the point pattern to a specific window.
    - :py:meth:`convert_to_spatstat_ppp`: Converts the point pattern into a ``spatstat.geom.ppp`` R object.
    - :py:meth:`plot`: Plots the point pattern.

.. note::

        **Typical usage**:
            - The class :py:class:`~structure_factor.structure_factor.StructureFactor` gets initialized using a :py:class:`~structure_factor.point_pattern.PointPattern`.

"""

import matplotlib.pyplot as plt
import numpy as np
from rpy2 import robjects
from spatstat_interface.interface import SpatstatInterface

from structure_factor.spatial_windows import AbstractSpatialWindow

#! pass on doc done (Diala)
class PointPattern(object):
    r"""Encapsulate a realization of a point process, the observation window, and the intensity of the underlying point process.

    Args:
        points (np.ndarray): :math:`n \times d` array collecting :math:`n` points in dimension :math:`d`, consisting of a realization of a point process.

        window (AbstractSpatialWindow): Observation window.

        intensity(float, optional): Intensity of the point process. If None, the intensity of the point process is approximated by the ratio of the points number to the window volume. Defaults to None.

    Keyword Args:
        params: Possible additional parameters of the point process.

    Example:
        .. plot:: code/point_pattern/point_pattern.py
            :include-source: True
            :align: center

    .. seealso::
            :py:mod:`~structure_factor.spatial_windows`,  :py:mod:`~structure_factor.point_processes`, :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :py:meth:`~structure_factor.point_pattern.PointPattern.plot`.
    """

    def __init__(self, points, window, intensity=None, **params):
        r"""Initialize the object from a realization ``points`` of the underlying point process with intensity ``intensity`` observed in ``window``.

        Args:
            points (np.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d` consisting of a realization of a point process.

            window (AbstractSpatialWindow, optional): Observation window containing the ``points``.

            intensity(float, optional): Intensity of the point process. If None, the intensity of the point process is approximated by the ratio of the number of point to the window volume. Defaults to None.

        Keyword Args:
            params: Possible additional parameters of the point process.

        """
        _points = np.array(points)
        assert _points.ndim == 2
        self.points = _points

        if window is not None:
            assert isinstance(window, AbstractSpatialWindow)
        self.window = window

        if intensity is not None:
            assert intensity > 0
        elif window is not None:
            intensity = self.points.shape[0] / window.volume
        self.intensity = intensity
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
            window (AbstractSpatialWindow): New observation window to restrict to.

        Returns:
            ~structure_factor.point_pattern.PointPattern: Restriction of the ``PointPattern`` to the prescribed ``window``.

        .. testsetup::

            from structure_factor.data import load_data #import data

        Example:
            .. plot:: code/point_pattern/restrict_pp.py
                :include-source: True
                :align: center

        .. seealso::
            :py:mod:`~structure_factor.spatial_windows`,  :py:mod:`~structure_factor.point_processes`, :py:meth:`~structure_factor.point_pattern.PointPattern.plot`.

        """
        assert isinstance(window, AbstractSpatialWindow)
        points = self.points[window.indicator_function(self.points)]
        return PointPattern(points, window, self.intensity)

    def convert_to_spatstat_ppp(self, **params):
        """Convert the object attributes :py:attr:`~structure_factor.point_pattern.PointPattern.points` and :py:attr:`~structure_factor.point_pattern.PointPattern.window` into a point pattern ``spatstat.geom.ppp`` R object.

        Keyword args:
            params (dict): Optional keyword arguments passed to ``spatstat.geom.ppp``.

        Returns:
            Point pattern R object of type ``spatstat.geom.ppp``.

        .. seealso::

            `https://rdrr.io/cran/spatstat.geom/man/ppp.html <https://rdrr.io/cran/spatstat.geom/man/ppp.html>`_.
        """
        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        window = params.get("window", self.window)
        if window is not None and isinstance(window, AbstractSpatialWindow):
            params["window"] = window.to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)

    def plot(self, axis=None, window_res=None, file_name="", **kwargs):
        """Display scatter plot of the attribute :py:attr:`~structure_factor.point_pattern.PointPattern.points`.

        Args:
            axis (matplotlib.axis, optional): Support axis of the plot. Defaults to None.

            window_res (AbstractSpatialWindow, optional): Output observation window. Defaults to None.

        Returns:
            matplotlib.axis: Plot axis.
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        if window_res is None:
            points = self.points
        else:
            assert isinstance(window_res, AbstractSpatialWindow)
            res_pp = self.restrict_to_window(window=window_res)
            points = res_pp.points

        kwargs.setdefault("c", "k")
        kwargs.setdefault("s", 0.5)
        axis.scatter(points[:, 0], points[:, 1], **kwargs)
        axis.set_aspect("equal", "box")

        if file_name:
            fig.savefig(file_name, bbox_inches="tight")
        return axis
