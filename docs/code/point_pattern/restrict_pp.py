import matplotlib.pyplot as plt

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow, BoxWindow

point_process = HomogeneousPoissonPointProcess(intensity=1)

window = BoxWindow([[-50, 50], [-50, 50]])
point_pattern = point_process.generate_point_pattern(window=window)

# Restrict to a BallWindow
ball = BallWindow(center=[0, 0], radius=30)
restricted_point_pattern = point_pattern.restrict_to_window(ball)

ax = restricted_point_pattern.plot()
ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
