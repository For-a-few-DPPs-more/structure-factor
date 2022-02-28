import matplotlib.pyplot as plt

from structure_factor.spatial_windows import BoxWindow

window = BoxWindow(bounds=[[-12, 10], [-20, 7]])
points = window.rand(n=400)

fig, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], "b.")
ax.set_aspect("equal", "box")
plt.tight_layout(pad=1)
