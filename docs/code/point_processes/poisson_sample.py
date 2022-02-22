import matplotlib.pyplot as plt

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

point_process = HomogeneousPoissonPointProcess(intensity=0.25)

window = BoxWindow([[-6, 6], [-6, 10], [-5, 6]])
points = point_process.generate_sample(window)

ax = plt.axes(projection="3d")
ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c="grey")

plt.tight_layout(pad=1)
