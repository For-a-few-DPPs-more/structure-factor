# Generate a PointPattern in a BoxWindow
from structure_factor.point_process import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.point_pattern import PointPattern

poisson = HomogeneousPoissonPointProcess(intensity=1)  # Initialize a Poisson point process
window = BallWindow(center=[0,0], radius=50) # Creat a ball window
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)