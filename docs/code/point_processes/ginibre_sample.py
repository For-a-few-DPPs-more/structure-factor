from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0, 0], radius=20)  # Observation window
ginibre = GinibrePointProcess()  # Ginibre process
ginibre_sample = ginibre.generate_sample(window=window)  # Sample of points

# Plot
import matplotlib.pyplot as plt

plt.plot(ginibre_sample[:, 0], ginibre_sample[:, 1], "k.")
plt.show()
