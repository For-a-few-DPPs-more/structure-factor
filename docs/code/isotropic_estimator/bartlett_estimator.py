# Generate a PointPattern in a BallWindow
import numpy as np

from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow

poisson = HomogeneousPoissonPointProcess(
    intensity=1 / (4 * np.pi)
)  # Initialize a Poisson point process
window = BallWindow(radius=20, center=[0, 0, 0])
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Point pattern

# Bartlett's estimator
from structure_factor.isotropic_estimator import bartlett_estimator

k_norm = np.linspace(1, 5, 3)  # wavenumbers
k_norm, estimation = bartlett_estimator(poisson_pp, k_norm=k_norm)
