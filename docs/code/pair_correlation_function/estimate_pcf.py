# Generate a PointPattern in a BoxWindow
from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow

poisson = HomogeneousPoissonPointProcess(
    intensity=1
)  # Initialize a Poisson point process
window = BallWindow(center=[0, 0], radius=40)  # Creat a ball window
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(points=poisson_points, window=window)  # Poisson PointPattern

# Approximate the pair correlation function
from structure_factor.pair_correlation_function import PairCorrelationFunction as pcf

poisson_pcf_fv = pcf.estimate(
    poisson_pp, method="fv", Kest=dict(rmax=20), fv=dict(method="c", spar=0.2)
)

# Plot the results
pcf.plot(
    poisson_pcf_fv,
    exact_pcf=poisson.pair_correlation_function,
    figsize=(7, 6),
    color=["grey"],
    style=["."],
)
