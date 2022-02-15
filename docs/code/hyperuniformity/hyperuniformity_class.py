# Load Ginibre PointPattern and restrict to BoxWindow
from structure_factor.data import load_data
import numpy as np
ginibre_pp = load_data.load_ginibre()

# Approximate the structure factor
from structure_factor.structure_factor import StructureFactor
sf_ginibre = StructureFactor(ginibre_pp)
k_norm, s_bi = sf_ginibre.bartlett_isotropic_estimator(n_allowed_k_norm=40) # Estimate the structure factor

# Hyperuniformity class
from structure_factor.hyperuniformity import Hyperuniformity
hyperuniformity_test = Hyperuniformity(k_norm, s_bi)
alpha_ginibre, _ = hyperuniformity_test.hyperuniformity_class(k_norm_stop=0.4)

# Visualization of the results
import matplotlib.pyplot as plt
import structure_factor.utils as utils
fitted_poly = hyperuniformity_test.fitted_poly # Fitted polynomial to s_bi
fig, axis =plt.subplots(figsize=(7,5))
axis.plot(k_norm, s_bi, 'b', marker=".", label="Approximated structure factor")
axis.plot(k_norm, utils.structure_factor_ginibre(k_norm), 'g', label=r"$S(k)$")
axis.plot(k_norm, fitted_poly(k_norm), 'r--', label= "Fitted line")
axis.annotate(r" $\alpha$ ={}".format(alpha_ginibre), xy=(0, 0), xytext=(0.01,0.1),
            arrowprops=dict(facecolor='black', shrink=0.0001))
axis.legend()
axis.set_xlabel('wavelength (k)')
axis.set_ylabel(r"Structure factor ($\mathsf{S}(k)$")
plt.show()