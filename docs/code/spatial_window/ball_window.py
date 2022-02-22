import matplotlib.pyplot as plt

from structure_factor.spatial_windows import BallWindow

window = BallWindow(radius=10, center=[0, 0])
points = window.rand(n=400)

fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], "b.")
ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
