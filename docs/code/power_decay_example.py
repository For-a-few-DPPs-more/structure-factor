import numpy as np

import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# approximating the structure factor
# create a box window
L = 70  # sidelength of the window
bounds = np.array([[-L / 2, L / 2], [-L / 2, L / 2]])  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window
# restrict the point pattern to box window
ginibre_pp_box = ginibre_pp.restrict_to_window(window)
# scattering intensity
sf_ginibre_box = StructureFactor(ginibre_pp_box)  # initialize the class StructureFactor
norm_k, si = sf_ginibre_box.scattering_intensity(k_max=6, meshgrid_shape=(200, 200))


# estimate the class of hyperuniformity 
from structure_factor.hyperuniformity import Hyperuniformity

# initialize Hyperuniformity
hyperuniformity_test = Hyperuniformity(norm_k, si)
# regularization of the approximated structure factor 
hyperuniformity_test.bin_data(bins=40)
# find the power decay of the structure factor
sf_power_decay, c = hyperuniformity_test.hyperuniformity_class(k_norm_stop=1)
print( "The estimated power of the decay to zero of the approximated structure factor is:",sf_power_decay)

import matplotlib.pyplot as plt

mean_k_norm = hyperuniformity_test.k_norm
mean_sf = hyperuniformity_test.sf
fitted_poly = hyperuniformity_test.fitted_poly
x = np.linspace(0, 2, 300)
y = np.linspace(0, 9, 500)
fig = plt.figure(figsize=(10, 6))
plt.plot(mean_k_norm, mean_sf, "b.", label="regularized sf")
plt.plot(mean_k_norm, mean_sf, "b")
plt.plot(y, utils.structure_factor_ginibre(y), "g", label="exact sf")
plt.plot(x, fitted_poly(x), "r--", label="fitted polynomial")
plt.legend()
plt.xlabel("wavelength ($||\mathbf{k}||$)")
plt.ylabel("Structure factor ($\mathsf{S}(\mathbf{k})$)")
plt.title("Test of hyperuniformity class of the Ginibre ensemble.")
plt.show()
