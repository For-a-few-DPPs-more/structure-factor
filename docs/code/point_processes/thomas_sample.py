import numpy as np

from structure_factor.point_processes import ThomasPointProcess
from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0, -10, 3], radius=10)  # Observation window
thomas = ThomasPointProcess(kappa=1 / (5 * np.pi), mu=5, sigma=1)  # Thomas process
thomas_sample = thomas.generate_sample(window=window)  # Sample of points

# Plot
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection="3d")
ax.scatter3D(
    thomas_sample[:, 0], thomas_sample[:, 1], thomas_sample[:, 2], c=thomas_sample[:, 2]
)
plt.show()
