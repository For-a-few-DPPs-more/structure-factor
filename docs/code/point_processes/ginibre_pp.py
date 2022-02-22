from structure_factor import point_pattern
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0, 0], radius=40)
point_process = GinibrePointProcess()
point_pattern = point_process.generate_point_pattern(window=window)
point_pattern.plot()
