from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

bounds = [[-6, 6], [-6, 10], [-5, 6]] # Window bounds
window = BoxWindow(bounds) # Observation window
poisson = HomogeneousPoissonPointProcess(intensity=1/4) # Poisson process
poisson_sample = poisson.generate_sample(window) # Sample of points

# Plot
import matplotlib.pyplot as plt
ax = plt.axes(projection='3d')
ax.scatter3D(poisson_sample[:, 0], poisson_sample[:, 1], poisson_sample[:, 2], c='grey')
plt.show()