from rpy2 import robjects
from structure_factor.spatial_windows import AbstractSpatialWindow
from structure_factor.spatstat_interface import SpatstatInterface


class PointPattern(object):
    def __init__(self, points, window=None):
        r"""[summary]

        Args:
            points (np.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d`.
            window (AbstractSpatialWindow, optional): Observation window containing the ``points``. Defaults to None.
        """
        self.points = points
        self.window = window

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
        window = params.setdefault("window", self.window)
        if isinstance(window, AbstractSpatialWindow):
            params["window"] = window.convert_to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)
