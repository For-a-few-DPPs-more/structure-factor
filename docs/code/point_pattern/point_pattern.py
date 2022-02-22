from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

point_process = HomogeneousPoissonPointProcess(intensity=1)

window = BoxWindow([[-50, 50], [-50, 50]])
points = point_process.generate_sample(window=window)
point_pattern = PointPattern(
    points=points, window=window, intensity=point_process.intensity
)
point_pattern.plot()
