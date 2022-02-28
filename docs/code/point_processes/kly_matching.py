import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from structure_factor.point_processes import (
    HomogeneousPoissonPointProcess,
    mutual_nearest_neighbor_matching,
)
from structure_factor.spatial_windows import BoxWindow
from structure_factor.utils import meshgrid_to_column_matrix

seed = None
rng = np.random.default_rng(seed)

bounds = np.array([[0, 20], [0, 20]])
window = BoxWindow(bounds)

# Perturbed grid
stepsize = 1.0
ranges = (np.arange(a, b, step=stepsize) for a, b in window.bounds)
X = meshgrid_to_column_matrix(np.meshgrid(*ranges))
shift = rng.uniform(0.0, stepsize, size=window.dimension)
X += shift

# Poisson
rho = (1 + 1) / (stepsize ** window.dimension)
ppp = HomogeneousPoissonPointProcess(rho)
Y = ppp.generate_sample(window, seed=rng)

# Mutual nearest neighbor matching
# To trigger periodic boundary conditions use boxsize
# make sure points belong to Π_i [0, L_i) (upper bound excluded)
boxsize = window.bounds[:, 1]  # default None
matching = mutual_nearest_neighbor_matching(X, Y, boxsize=boxsize)

# Display
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(X[:, 0], X[:, 1], s=5, c="blue", label="grid")
ax.scatter(Y[:, 0], Y[:, 1], s=0.4, c="orange", label="Poisson")
ax.scatter(Y[matching, 0], Y[matching, 1], s=5, c="red", label="Poisson matched")
lines = LineCollection(
    [[x, y] for x, y in zip(X, Y[matching])], color="gray", label="matchings"
)
ax.add_collection(lines)

ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
