from structure_factor.homogeneous_poisson_process import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

bounds = [[-6, 6], [-6, 6]]
window = BoxWindow(bounds)
poisson = HomogeneousPoissonPointProcess(intensity=1)
poisson_sample = poisson.generate_sample(window)

# plot
import matplotlib.pyplot as plt

plt.plot(poisson_sample[:, 0], poisson_sample[:, 1], "k.")
plt.show()
