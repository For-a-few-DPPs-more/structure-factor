# Generate a Thomas PointPattern
from structure_factor.point_processes import ThomasPointProcess
from structure_factor.spatial_windows import BallWindow
import numpy as np

window = BallWindow(center=[-20,-10], radius=50) # Observation window
thomas = ThomasPointProcess(kappa=1/(20*np.pi), mu= 20, sigma=2) # Thomas process
thomas_pp = thomas.generate_point_pattern(window= window) # PointPattern
thomas_pp.plot() # Plot