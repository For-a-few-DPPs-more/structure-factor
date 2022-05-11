import matplotlib.pyplot as plt
import numpy as np
from structure_factor.data import load_data
from structure_factor.hyperuniformity import effective_hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)

point_pattern = load_data.load_ginibre()
point_process = GinibrePointProcess()

sf = StructureFactor(point_pattern)
d, r = point_pattern.dimension, point_pattern.window.radius
k_norm = allowed_k_norm_bartlett_isotropic(dimension=d, radius=r, nb_values=60)
k_norm, sf_estimated = sf.bartlett_isotropic_estimator(k_norm)

summary = effective_hyperuniformity(k_norm, sf_estimated, k_norm_stop=0.2)
x = np.linspace(0, 1, 300)
sf_fitted_line = summary["fitted_line"](x)

sf_theoretical = point_process.structure_factor(k_norm)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(k_norm, sf_theoretical, "g", label=r"$S(\mathbf{k})$")
ax.plot(k_norm, sf_estimated, "b", marker=".", label="Approximated structure factor")
ax.plot(x, sf_fitted_line, "r--", label="Fitted line")

ax.annotate(
    "H={}".format(summary["H"]),
    xy=(0, 0),
    xytext=(0.01, 0.1),
    arrowprops=dict(facecolor="black", shrink=0.0001),
)
ax.legend()
ax.set_xlabel(r"$||\mathbf{k}||_2$")
ax.set_ylabel(r"$\mathsf{S}(\mathbf{k})$")
plt.tight_layout(pad=1)
