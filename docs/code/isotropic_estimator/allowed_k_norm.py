from structure_factor.isotropic_estimator import allowed_k_norm

k = allowed_k_norm(dimension=2, radius=20, nb_values=6)
print(k)
