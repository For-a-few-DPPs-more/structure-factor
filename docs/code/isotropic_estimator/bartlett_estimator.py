import numpy as np

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.tapered_estimators_isotropic import bartlett_estimator

point_process = HomogeneousPoissonPointProcess(intensity=1 / (4 * np.pi))
window = BallWindow(radius=20, center=[0, 0, 0])
point_pattern = point_process.generate_point_pattern(window=window)

k_norm = np.linspace(1, 5, 3)
k_norm, estimation = bartlett_estimator(k_norm, point_pattern)
