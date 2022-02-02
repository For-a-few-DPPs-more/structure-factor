# Generate a PointPattern in a BoxWindow
from structure_factor.point_process import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.point_pattern import PointPattern

poisson = HomogeneousPoissonPointProcess(intensity=1)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]])
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)

# Compute the scattering intensity on allowed wavevectors
k, s_si = sf_poisson.scattering_intensity(k_max=4)

# Visualize the result
import matplotlib.pyplot as plt

sf_poisson.plot_spectral_estimator(
    k, s_si, plot_type="all", exact_sf=poisson.structure_factor,
    error_bar=True, bins=30, label=r"$\widehat{S}_{\mathrm{SI}}(\mathbf{k})$")
plt.show()
