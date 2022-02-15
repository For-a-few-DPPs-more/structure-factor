from structure_factor.spatial_windows import BoxWindow
import matplotlib.pyplot as plt

window = BoxWindow(bounds=[[-12, 10], [-20, 7]])
points = window.rand(n=400)
plt.plot(points[:,0], points[:,1], 'b.')
plt.show()