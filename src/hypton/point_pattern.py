import matplotlib.pyplot as plt
from rpy2 import robjects

from hypton.spatial_windows import AbstractSpatialWindow
from hypton.spatstat_interface import SpatstatInterface


class PointPattern(object):
    """Implementation of a new object associated to a point process (or a configuration of points). This class will contains the a sample of points of the point process, the window containing the sample and the intensity of the point process.

    .. note::

        Typical usage:
            -The class :py:class:`~.hypton.StructureFactor` take an object of type ``PointPattern``.

            -Convert Python :py:class:`PointPattern` object to `spatstat <https://cran.r-project.org/web/packages/spatstat/index.html#:~:text=spatstat%3A%20Spatial%20Point%20Pattern%20Analysis,for%20analysing%20Spatial%20Point%20Patterns.&text=Contains%20over%202000%20functions%20for,model%20diagnostics%2C%20and%20formal%20inference>`_ point pattern R object.
    """

    def __init__(self, points, window=None, intensity=None):
        r"""

        Args:
            points (np.ndarray): :math:`N \times d` array collecting :math:`N` points in dimension :math:`d` consisting of a realization of a point process.

            window (AbstractSpatialWindow, optional): Observation window containing the ``points``. Defaults to None.

            intensity(float, optional): intensity of the point process. Defaults to None.
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
        """Ambient dimension of the space where the points live"""
        return self.points.shape[1]

    def restrict_to_window(self, window):
        """Restrict the realization of point process to a window of type :py:class:`~.spatial_windows.AbstractSpatialWindow`

        Args:
            window (AbstractSpatialWindow): :py:class:`~.spatial_windows.BallWindow` or :py:class:`~.spatial_windows.BoxWindow`.

        Returns:
             restriction of the ``PointPattern`` object to the specifyied window

        """
        assert isinstance(window, AbstractSpatialWindow)
        points = self.points[window.indicator_function(self.points)]
        return PointPattern(points, window, self.intensity)

    def convert_to_spatstat_ppp(self, **params):
        """Convert the object attributes :py:attr:`.points` and :py:attr:`.window` into a point pattern ``spatstat.geom.ppp`` R object.
        ``params`` corresponds to optional keyword arguments passed to ``spatstat.geom.ppp``.

        Returns:
            Point pattern R object of type ``spatstat.geom.ppp``.
        """

        spatstat = SpatstatInterface(update=False)
        spatstat.import_package("geom", update=False)
        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        window = params.get("window", self.window)
        if window is not None and isinstance(window, AbstractSpatialWindow):
            params["window"] = window.convert_to_spatstat_owin()
        return spatstat.geom.ppp(x, y, **params)

    def plot(self, axis=None, window_res=None, c="k,", file_name=""):
        """Visualization of the plot of ``PointPattern.points``.

        Args:
            axis ([axis, optional): support axis of the plot. Defaults to None.

            window_res (AbstractSpatialWindow, optional): window used to visualized the plot. Defaults to None.

        Returns:
            plot of ``PointPattern.points`` (in the restricted window window_res if specified).
        """
        if axis is None:
            fig, axis = plt.subplots(figsize=(5, 5))

        if window_res is None:
            points = self.points
        else:
            assert isinstance(window_res, AbstractSpatialWindow)
            res_pp = self.restrict_to_window(window=window_res)
            points = res_pp.points

        axis.plot(points[:, 0], points[:, 1], c)
        axis.set_aspect("equal", "box")

        if file_name:
            fig.savefig(file_name, bbox_inches="tight")
        return axis
