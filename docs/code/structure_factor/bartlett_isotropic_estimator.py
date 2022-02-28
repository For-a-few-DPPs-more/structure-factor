import matplotlib.pyplot as plt

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)

point_process = HomogeneousPoissonPointProcess(intensity=1)
window = BallWindow(center=[0, 0], radius=50)
point_pattern = point_process.generate_point_pattern(window=window)

sf = StructureFactor(point_pattern)
d, r = point_pattern.dimension, point_pattern.window.radius
k_norm = allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=60)
k_norm, sf_estimated = sf.bartlett_isotropic_estimator(k_norm)

ax = sf.plot_isotropic_estimator(
    k_norm, sf_estimated, label=r"$\widehat{S}_{\mathrm{BI}}(k)$"
)

plt.tight_layout(pad=1)
