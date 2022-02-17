# Generate a Poisson PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

window = BoxWindow(bounds=[[-2, 20], [3, 15]]) # Observation window
poisson = HomogeneousPoissonPointProcess(intensity=6) # Poisson process
poisson_pp = poisson.generate_point_pattern(window=window) # PointPattern
poisson_pp.plot() # Plot