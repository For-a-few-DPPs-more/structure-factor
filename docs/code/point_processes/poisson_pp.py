from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

point_process = HomogeneousPoissonPointProcess(intensity=6)

window = BoxWindow(bounds=[[-2, 20], [3, 15]])
point_pattern = point_process.generate_point_pattern(window=window)

point_pattern.plot()
