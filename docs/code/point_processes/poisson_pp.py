import matplotlib.pyplot as plt

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

point_process = HomogeneousPoissonPointProcess(intensity=6)

window = BoxWindow(bounds=[[-10, 10], [-10, 10]])
point_pattern = point_process.generate_point_pattern(window=window)

ax = point_pattern.plot()
ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
