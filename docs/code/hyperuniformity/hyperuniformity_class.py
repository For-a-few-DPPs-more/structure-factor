import matplotlib.pyplot as plt

from structure_factor.data import load_data
from structure_factor.hyperuniformity import Hyperuniformity
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.structure_factor import StructureFactor

point_process = GinibrePointProcess()
point_pattern = load_data.load_ginibre()
sf = StructureFactor(point_pattern)
k_norm, sf_estimated = sf.bartlett_isotropic_estimator(n_allowed_k_norm=40)

sf_theoretical = point_process.structure_factor(k_norm)

hyperuniformity = Hyperuniformity(k_norm, sf_estimated)
alpha, _ = hyperuniformity.hyperuniformity_class(k_norm_stop=0.4)
sf_fitted_0 = hyperuniformity.fitted_poly(k_norm)

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(k_norm, sf_theoretical, "g", label=r"$S(k)$")
ax.plot(k_norm, sf_estimated, "b", marker=".", label="Approximated structure factor")
ax.plot(k_norm, sf_fitted_0, "r--", label="Fitted line")

ax.annotate(
    r"$\alpha$ ={}".format(alpha),
    xy=(0, 0),
    xytext=(0.01, 0.1),
    arrowprops=dict(facecolor="black", shrink=0.0001),
)
ax.legend()
ax.set_xlabel("wavelength (k)")
ax.set_ylabel(r"Structure factor ($\mathsf{S}(k)$")

plt.tight_layout(pad=1)
