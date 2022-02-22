import matplotlib.pyplot as plt
import numpy as np

from structure_factor.data import load_data
from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.structure_factor import StructureFactor

ginibre_pp = load_data.load_ginibre()

point_pattern = load_data.load_ginibre()
sf = StructureFactor(point_pattern)
k_norm, sf_estimated = sf.bartlett_isotropic_estimator(n_allowed_k_norm=50)

sf_theoretical = GinibrePointProcess.structure_factor(k_norm)

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
plt.show()
