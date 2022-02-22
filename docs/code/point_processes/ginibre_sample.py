import matplotlib.pyplot as plt

from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow

point_process = GinibrePointProcess()

window = BallWindow(center=[0, 0], radius=20)
points = point_process.generate_sample(window=window)

plt.plot(points[:, 0], points[:, 1], "k.")
plt.show()
