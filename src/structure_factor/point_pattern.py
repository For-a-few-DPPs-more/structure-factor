from rpy2 import robjects
from structure_factor.spatial_windows import AbstractSpatialWindow
from structure_factor.spatstat_interface import SpatstatInterface
from structure_factor.spatial_windows import BoxWindow
import numpy as np


class PointPattern(object):
    def __init__(self, points, window=None, intensity=None):
        r"""[summary]

        Args:
            points (np.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d`.
            window (AbstractSpatialWindow, optional): Observation window containing the ``points``. Defaults to None.
        """
        self.points = points
        self.window = window
        self.intensity = intensity

    def dimension(self):
        """Ambient dimension where the points live"""
        return self.points.shape[1]

    def convert_to_spatstat_ppp(self, **params):
        """Convert Python :py:class:`PointPattern` object to ``spatstat`` point pattern R object, using ``spatstat.geom.ppp``

        Returns:
            [type]: spatstat point pattern
        """

        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        window = params.get("window", self.window)
        if window is not None and isinstance(window, AbstractSpatialWindow):
            params["window"] = window.convert_to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)

    def restrict_to_cubic_window(self, x_min, y_min, L):
        points = self.points
        index_x_in_cube = np.logical_and(
            x_min < points[:, 0],
            points[:, 0] < x_min + L,
        )
        bounds = np.array([[x_min, y_min], [x_min + L, y_min + L]])
        index_y_in_cube = np.logical_and(y_min < points[:, 1], points[:, 1] < y_min + L)
        index_points_in_cube = np.logical_and(index_x_in_cube, index_y_in_cube)
        points_in_cube = points[index_points_in_cube]
        window = BoxWindow(bounds)
        return PointPattern(points_in_cube, window, self.intensity)
