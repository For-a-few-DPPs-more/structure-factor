# Load Ginibre PointPattern and restrict to BoxWindow
import numpy as np

from structure_factor.data import load_data

ginibre_pp = load_data.load_ginibre()

# Approximate the structure factor
from structure_factor.structure_factor import StructureFactor

sf_ginibre = StructureFactor(ginibre_pp)
k_norm, s_bi = sf_ginibre.bartlett_isotropic_estimator(
    n_allowed_k_norm=50
)  # Estimate the structure factor

# Effective hyperuniformity
from structure_factor.hyperuniformity import Hyperuniformity

hyperuniformity_test = Hyperuniformity(k_norm, s_bi)
H_ginibre, _ = hyperuniformity_test.effective_hyperuniformity(k_norm_stop=0.2)

# Visualization of the results
import matplotlib.pyplot as plt

from structure_factor.point_processes import GinibrePointProcess

fitted_line = hyperuniformity_test.fitted_line  # Fitted line to s_bi
x = np.linspace(0, 1, 300)
fig, axis = plt.subplots(figsize=(7, 5))
axis.plot(k_norm, s_bi, "b", marker=".", label="Approximated structure factor")
axis.plot(x, fitted_line(x), "r--", label="Fitted line")
axis.plot(k_norm, GinibrePointProcess.structure_factor(k_norm), "g", label=r"$S(k)$")
axis.annotate(
    "H={}".format(H_ginibre),
    xy=(0, 0),
    xytext=(0.01, 0.1),
    arrowprops=dict(facecolor="black", shrink=0.0001),
)
axis.legend()
axis.set_xlabel("wavelength (k)")
axis.set_ylabel(r"Structure factor ($\mathsf{S}(k)$")
plt.show()
