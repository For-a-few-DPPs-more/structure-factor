import matplotlib.pyplot as plt
import numpy as np

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapers import SineTaper

point_process = HomogeneousPoissonPointProcess(intensity=1 / np.pi)

window = BoxWindow([[-50, 50], [-50, 50]])
point_pattern = point_process.generate_point_pattern(window=window)

sf = StructureFactor(point_pattern)

x = np.linspace(-3, 3, 100)
x = x[x != 0]
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel()))

taper = SineTaper([1, 1])
sf_estimated = sf.tapered_estimator(k=k, taper=taper, debiased=True, direct=True)

sf.plot_tapered_estimator(
    k,
    sf_estimated,
    plot_type="all",
    exact_sf=point_process.structure_factor,
    error_bar=True,
    bins=30,
    label=r"$\widehat{S}_{\mathrm{DDTP}}(t_1, \mathbf{k})$",
)

plt.tight_layout(pad=1)
