# Generate a Poisson PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.point_pattern import PointPattern

poisson = HomogeneousPoissonPointProcess(
    intensity=1
)  # Initialize a Poisson point process
window = BallWindow(center=[0, 0], radius=50)  # Creat a ball window
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)


# Compute Bartlett isotropic estimator
k_norm, s_bi = sf_poisson.bartlett_isotropic_estimator(
    n_allowed_k_norm=50
)  # on allowed wavenumbers

# Visualize the result
import matplotlib.pyplot as plt

sf_poisson.plot_isotropic_estimator(
    k_norm, s_bi, label=r"$\widehat{S}_{\mathrm{BI}}(k)$"
)
plt.show()
