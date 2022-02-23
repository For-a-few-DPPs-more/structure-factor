from structure_factor.tapered_estimators_isotropic import allowed_k_norm

k = allowed_k_norm(dimension=2, radius=20, nb_values=6)
print(k)
