import matplotlib.pyplot as plt
import numpy as np

from structure_factor.point_processes import ThomasPointProcess
from structure_factor.spatial_windows import BallWindow

point_process = ThomasPointProcess(kappa=1 / (5 * np.pi), mu=5, sigma=1)

window = BallWindow(center=[0, -10, 3], radius=10)
points = point_process.generate_sample(window=window)

fig = plt.figure(figsize=(9, 6))
ax = plt.axes(projection="3d")
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2])

plt.tight_layout(pad=1)
