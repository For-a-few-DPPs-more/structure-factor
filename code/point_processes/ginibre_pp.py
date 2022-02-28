import matplotlib.pyplot as plt

from structure_factor import point_pattern
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0, 0], radius=40)
point_process = GinibrePointProcess()
point_pattern = point_process.generate_point_pattern(window=window)

ax = point_pattern.plot()
ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
