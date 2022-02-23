import matplotlib.pyplot as plt
import numpy as np

from structure_factor.data import load_data
from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapered_estimators_isotropic import allowed_k_norm

point_pattern = load_data.load_ginibre()
point_process = GinibrePointProcess()

sf = StructureFactor(point_pattern)
d, r = point_pattern.dimension, point_pattern.window.radius
k_norm = allowed_k_norm(dimension=d, radius=r, nb_values=60)
k_norm, sf_estimated = sf.tapered_estimator_isotropic(k_norm)

sf_theoretical = point_process.structure_factor(k_norm)

hyperuniformity = Hyperuniformity(k_norm, sf_estimated)
H_ginibre, _ = hyperuniformity.effective_hyperuniformity(k_norm_stop=0.2)
x = np.linspace(0, 1, 300)
sf_fitted_line = hyperuniformity.fitted_line(x)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(k_norm, sf_theoretical, "g", label=r"$S(k)$")
ax.plot(k_norm, sf_estimated, "b", marker=".", label="Approximated structure factor")
ax.plot(x, sf_fitted_line, "r--", label="Fitted line")

ax.annotate(
    "H={}".format(H_ginibre),
    xy=(0, 0),
    xytext=(0.01, 0.1),
    arrowprops=dict(facecolor="black", shrink=0.0001),
)
ax.legend()
ax.set_xlabel("wavelength (k)")
ax.set_ylabel(r"Structure factor ($\mathsf{S}(k)$")

plt.tight_layout(pad=1)
