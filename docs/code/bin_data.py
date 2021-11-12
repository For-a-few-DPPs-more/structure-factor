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
k_norm, sf = sf_ginibre_box.scattering_intensity(k_max=6, meshgrid_shape=(200, 200))


# test effective hyperuniformity
from structure_factor.hyperuniformity import Hyperuniformity

# initialize Hyperuniformity
hyperuniformity_test = Hyperuniformity(k_norm, sf)
# regularization of sf
hyperuniformity_test.bin_data(bins=60)

# plot the results of bin_data method
import matplotlib.pyplot as plt
import numpy as np
_, axis = plt.subplots(figsize=(8, 6))
utils.plot_approximation(k_norm,
                         sf, axis, 
                         label="approximated sf", 
                         color="grey",linestyle="",
                         marker=".",
                         markersize=1.5,)
utils.plot_summary(k_norm, sf, axis, bins=60)
utils.plot_exact(k_norm, utils.structure_factor_ginibre, axis, label="exact sf")
axis.legend()
axis.title.set_text("Regularization of the approximated structure factor of the Ginibre ensemble")
