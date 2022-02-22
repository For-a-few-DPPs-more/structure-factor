# Generate PointPattern of a Poisson Process
from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow

poisson = HomogeneousPoissonPointProcess(
    intensity=1
)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]])
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern
poisson_pp.plot()
