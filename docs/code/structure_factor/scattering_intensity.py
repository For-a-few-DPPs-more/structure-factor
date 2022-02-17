# Generate a PointPattern in a BoxWindow
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
import numpy as np

# Generate a Poisson PointPattern
poisson = HomogeneousPoissonPointProcess(intensity=1/np.pi)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]]) # Observation window
poisson_pp = poisson.generate_point_pattern(window=window) # PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)

# Compute the scattering intensity on allowed wavevectors
k, s_si = sf_poisson.scattering_intensity(k_max=4)

# Visualize the result
import matplotlib.pyplot as plt
sf_poisson.plot_spectral_estimator(
    k, s_si, plot_type="all", error_bar=True, bins=30, exact_sf=poisson.structure_factor, 
    scale="log",
    label=r"$\widehat{S}_{\mathrm{SI}}(\mathbf{k})$"
)
plt.show()