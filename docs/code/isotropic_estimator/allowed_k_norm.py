from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
)

k = allowed_k_norm_bartlett_isotropic(dimension=2, radius=20, nb_values=6)
print(k)
