import matplotlib.pyplot as plt
import numpy as np

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapers import multi_sinetaper_grid

point_process = HomogeneousPoissonPointProcess(intensity=1)
window = BoxWindow([[-50, 50], [-50, 50]])
point_pattern = point_process.generate_point_pattern(window=window)

sf = StructureFactor(point_pattern)

# Use the family of sine tapers (default)
x = np.linspace(-2, 2, 80)
x = x[x != 0]
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel()))

tapers = multi_sinetaper_grid(point_pattern.dimension, p_component_max=2)
sf_estimated = sf.tapered_estimator(k, tapers=tapers, debiased=True, direct=True)

sf.plot_non_isotropic_estimator(
    k,
    sf_estimated,
    plot_type="all",
    error_bar=True,
    bins=30,
    label=r"$\widehat{S}_{\mathrm{MDDTP}}((t_j)_1^4, \mathbf{k})$",
)

plt.tight_layout(pad=1)
