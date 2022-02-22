import matplotlib.pyplot as plt
import numpy as np

import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

ginibre_pp = load_data.load_ginibre()

# Restrict point pattern to smaller window
window = BoxWindow([[-35, 35], [-35, 35]])
ginibre_pp_box = ginibre_pp.restrict_to_window(window)

# Estimated the structure factor on a grid of wavevectors
sf = StructureFactor(ginibre_pp_box)
x = np.linspace(0, 3, 80)
x = x[x != 0]
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel()))
sf_estimated = sf.multitapered_periodogram(k)

k_norm = utils.norm_k(k)
hyperuniformity = Hyperuniformity(k_norm, sf_estimated)
k_norm_binned, sf_estimated_binned, _ = hyperuniformity.bin_data(bins=40)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(k_norm, sf_estimated, "b,", label="Before regularization", rasterized=True)
sf.plot_isotropic_estimator(
    k_norm_binned,
    sf_estimated_binned,
    axis=ax,
    color="m",
    exact_sf=GinibrePointProcess.structure_factor,
    label="After regularization",
)

ax.legend()
plt.show()
