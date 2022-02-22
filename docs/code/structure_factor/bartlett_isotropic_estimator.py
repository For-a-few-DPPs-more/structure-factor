import matplotlib.pyplot as plt

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.structure_factor import StructureFactor

point_process = HomogeneousPoissonPointProcess(intensity=1)
window = BallWindow(center=[0, 0], radius=50)
point_pattern = point_process.generate_point_pattern(window=window)

sf = StructureFactor(point_pattern)
k_norm, sf_estimated = sf.bartlett_isotropic_estimator(n_allowed_k_norm=50)

ax = sf.plot_isotropic_estimator(
    k_norm, sf_estimated, label=r"$\widehat{S}_{\mathrm{BI}}(k)$"
)

plt.tight_layout(pad=1)
