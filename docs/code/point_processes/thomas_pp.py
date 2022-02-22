import numpy as np

from structure_factor.point_processes import ThomasPointProcess
from structure_factor.spatial_windows import BallWindow

point_process = ThomasPointProcess(kappa=1 / (20 * np.pi), mu=20, sigma=2)

window = BallWindow(center=[-20, -10], radius=50)
point_pattern = point_process.generate_point_pattern(window=window)

point_pattern.plot()
