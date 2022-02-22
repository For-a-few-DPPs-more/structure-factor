import matplotlib.pyplot as plt

from structure_factor.spatial_windows import BallWindow

window = BallWindow(radius=10, center=[0, 0])
points = window.rand(n=400)
plt.plot(points[:, 0], points[:, 1], "b.")
plt.show()
