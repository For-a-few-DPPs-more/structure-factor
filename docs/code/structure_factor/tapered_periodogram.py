# Generate a PointPattern in a BoxWindow
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
import numpy as np


# Generate a Poisson PointPattern
poisson = HomogeneousPoissonPointProcess(intensity=1/np.pi)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]]) # Observation window
poisson_pp = poisson.generate_point_pattern(window=window)  # PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)

# Generalte wavevectors 
x = np.linspace(-3, 3, 100)
x = x[x != 0] # Get rid of zero
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel()))

# chose a taper
from structure_factor.tapers import SineTaper

p = [1, 1]
taper = SineTaper(p) # First taper of the sinusoidal tapers

# Compute the scaled tapered periodogram directly debiased
s_ddtp = sf_poisson.tapered_periodogram(k=k, taper=taper, debiased=True, direct=True)

# Visualize the result
import matplotlib.pyplot as plt

sf_poisson.plot_spectral_estimator(
    k, s_ddtp, plot_type="all", exact_sf=poisson.structure_factor, error_bar=True, 
    bins=30, label=r"$\widehat{S}_{\mathrm{DDTP}}(t_1, \mathbf{k})$")
plt.show()
