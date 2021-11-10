import numpy as np

from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# Approximating the structure factor
# creat box window
L = 70  # sidelength of the window
bounds = np.array([[-L / 2, L / 2], [-L / 2, L / 2]])  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window
# restrict to box window
ginibre_pp_box = ginibre_pp.restrict_to_window(window)
# scattering intensity
sf_ginibre_box = StructureFactor(ginibre_pp_box)  # initialize the class StructureFactor
norm_k, si = sf_ginibre_box.scattering_intensity(k_component_max=6, meshgrid_shape=200)


# test hyperuniformity class
from structure_factor.hyperuniformity import Hyperuniformity

# initialize Hyperuniformity
hyperuniformity_test = Hyperuniformity(norm_k, si)
# regularization of the approximated result
hyperuniformity_test.bin_data(bins=40)
# find power decay
sf_power_decay, c = hyperuniformity_test.hyperuniformity_class(k_norm_stop=1)
print(
    "The estimated power of the decay to zero of the approximated structure factor is:",
    sf_power_decay,
)
