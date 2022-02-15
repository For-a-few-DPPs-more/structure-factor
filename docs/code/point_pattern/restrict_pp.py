# Generate PointPattern of a Poisson Process in a box window
from structure_factor.point_process import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow, BallWindow
from structure_factor.point_pattern import PointPattern

poisson = HomogeneousPoissonPointProcess(intensity=1)  # Initialize a Poisson point process
window = BoxWindow([[-50, 50], [-50, 50]])
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Restrict to a BallWindow 
poisson_pp_ball = poisson_pp.restrict_to_window(BallWindow(radius=30, center=[0,0]))
poisson_pp_ball.plot()