from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor
import structure_factor.utils as utils

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# Approximating the structure factor
# creat box window
L = 70  # sidelength of the window
bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window
# restrict to box window
ginibre_pp_box = ginibre_pp.restrict_to_window(window)
# scattering intensity
sf_ginibre_box = StructureFactor(ginibre_pp_box)  # initialize the class StructureFactor
k_norm, si = sf_ginibre_box.scattering_intensity(k_max=6, meshgrid_shape=(200, 200))


# test effective hyperuniformity
from structure_factor.hyperuniformity import Hyperuniformity

# initialize Hyperuniformity
hyperuniformity_test = Hyperuniformity(k_norm, si)
# regularization of the approximated result
hyperuniformity_test.bin_data(bins=40)
# find the H index
H_ginibre, std = hyperuniformity_test.effective_hyperuniformity(k_norm_stop=4)
print("H_ginibre=", H_ginibre)


# plot
import matplotlib.pyplot as plt
import numpy as np

fitted_sf_line = hyperuniformity_test.fitted_line  # fittend ligne
index_peak = hyperuniformity_test.i_first_peak


mean_k_norm = hyperuniformity_test.k_norm
mean_sf = hyperuniformity_test.sf
x = np.linspace(0, 5, 300)
y = np.linspace(0, 15, 500)
fig = plt.figure(figsize=(10, 6))
plt.plot(mean_k_norm, mean_sf, "b.", label="approx_sf")
plt.plot(mean_k_norm, mean_sf, "b", label="approx_sf")
plt.plot(x, fitted_sf_line(x), "r--", label="fitted line")
plt.plot(y, utils.structure_factor_ginibre(y), "g", label="exact sf")
plt.plot(mean_k_norm[index_peak], mean_sf[index_peak], "k*", label="first peak")
plt.legend()
plt.xlabel("wavelength ($||\mathbf{k}||$)")
plt.ylabel("Structure factor ($\mathsf{S}(\mathbf{k})$)")
plt.show()
