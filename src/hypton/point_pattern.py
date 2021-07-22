from rpy2 import robjects
import matplotlib.pyplot as plt

from hypton.spatstat_interface import SpatstatInterface
from hypton.spatial_windows import AbstractSpatialWindow


class PointPattern(object):
    def __init__(self, points, window=None, intensity=None):
        r"""[summary]

        Args:
            points (np.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d`.
            window (AbstractSpatialWindow, optional): Observation window containing the ``points``. Defaults to None.
        """
        assert points.ndim == 2
        self.points = points

        if window is not None:
            assert isinstance(window, AbstractSpatialWindow)
        self.window = window

        if intensity is not None:
            assert intensity > 0
        elif window is not None:
            intensity = points.shape[0] / window.volume

        self.intensity = intensity

    @property
    def dimension(self):
        """Ambient dimension where the points live"""
        return self.points.shape[1]

    def restrict_to_window(self, window):
        assert isinstance(window, AbstractSpatialWindow)
        points = self.points[window.indicator_function(self.points)]
        return PointPattern(points, window, self.intensity)

    def convert_to_spatstat_ppp(self, **params):
        """Convert Python :py:class:`PointPattern` object to ``spatstat`` point pattern R object, using ``spatstat.geom.ppp``.
        This method converts the first two dimensions of the ``PointPattern.points`` into a ``spatstat.geom.ppp`` object.

        Returns:
            [type]: ``spatstat.geom.ppp`` point pattern
        """

        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        window = params.get("window", self.window)
        if window is not None and isinstance(window, AbstractSpatialWindow):
            params["window"] = window.convert_to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)

    def plot(self, axis=None):
        if axis is None:
            _, axis = plt.subplots(figsize=(5, 5))
        axis.plot(self.points[:, 0], self.points[:, 1], "k,")
        axis.set_aspect("equal", "box")
        return axis
