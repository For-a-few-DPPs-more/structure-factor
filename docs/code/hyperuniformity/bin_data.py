import matplotlib.pyplot as plt
import numpy as np

import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

point_pattern = load_data.load_ginibre()
point_process = GinibrePointProcess()

# Restrict point pattern to smaller window
window = BoxWindow([[-35, 35], [-35, 35]])
point_pattern_box = point_pattern.restrict_to_window(window)

# Estimated the structure factor on a grid of wavevectors
sf = StructureFactor(point_pattern_box)
x = np.linspace(0, 3, 80)
x = x[x != 0]
k = utils.meshgrid_to_column_matrix(np.meshgrid(x, x))
k, sf_estimated = sf.scattering_intensity(k)

k_norm = utils.norm(k)
hyperuniformity = Hyperuniformity(k_norm, sf_estimated)
k_norm_binned, sf_estimated_binned, _ = hyperuniformity.bin_data(bins=40)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(k_norm, sf_estimated, "b,", label="Before regularization", rasterized=True)
sf.plot_isotropic_estimator(
    k_norm_binned,
    sf_estimated_binned,
    axis=ax,
    color="m",
    exact_sf=point_process.structure_factor,
    label="After regularization",
)
ax.legend()

plt.tight_layout(pad=1)
