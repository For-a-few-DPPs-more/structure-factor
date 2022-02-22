# Generate a PointPattern in a BoxWindow
from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

poisson = HomogeneousPoissonPointProcess(
    intensity=1
)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]])
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)

# Generalte wavevectors
import numpy as np

x = np.linspace(-2, 2, 80)
x = x[x != 0]  # Get rid of zero
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel()))


# Compute the scaled multitapered periodogram undirectly debiased
s_mddtp = sf_poisson.multitapered_periodogram(
    k, debiased=True, direct=True, p_component_max=2
)  # Use the family of sine tapers (default)

# Visualize the result
import matplotlib.pyplot as plt

sf_poisson.plot_spectral_estimator(
    k,
    s_mddtp,
    plot_type="all",
    error_bar=True,
    bins=30,
    label=r"$\widehat{S}_{\mathrm{MDDTP}}((t_j)_1^4, \mathbf{k})$",
)
plt.show()
