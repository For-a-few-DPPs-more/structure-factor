import matplotlib.pyplot as plt
import numpy as np

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

point_process = HomogeneousPoissonPointProcess(intensity=1 / np.pi)

window = BoxWindow([[-50, 50], [-50, 50]])
point_pattern = point_process.generate_point_pattern(window=window)

sf = StructureFactor(point_pattern)
k, sf_estimated = sf.scattering_intensity(k_max=4)

sf.plot_non_isotropic_estimator(
    k,
    sf_estimated,
    plot_type="all",
    error_bar=True,
    bins=30,
    exact_sf=point_process.structure_factor,
    scale="log",
    label=r"$\widehat{S}_{\mathrm{SI}}(\mathbf{k})$",
)

plt.tight_layout(pad=1)
