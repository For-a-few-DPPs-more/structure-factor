# Generate a PointPattern in a BoxWindow
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.point_pattern import PointPattern

poisson = HomogeneousPoissonPointProcess(
    intensity=1
)  # Initialize a Poisson point process
window = BallWindow(center=[0, 0], radius=40)  # Creat a ball window
poisson_points = poisson.generate_sample(window=window)  # Sample a realization
poisson_pp = PointPattern(
    points=poisson_points, window=window, seed=1
)  # Poisson PointPattern

# Approximate the pair correlation function
from structure_factor.pair_correlation_function import PairCorrelationFunction as pcf

poisson_pcf_fv = pcf.estimate(
    poisson_pp, method="fv", Kest=dict(rmax=20), fv=dict(method="c", spar=0.2)
)

# Interpolate/extrapolate the results
r = poisson_pcf_fv["r"]
pcf_r = poisson_pcf_fv["pcf"]
poisson_pcf_fct = pcf.interpolate(
    r=r,
    pcf_r=pcf_r,
    drop=True,
)

# Plots the results
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 50, 200)
plt.plot(x, poisson_pcf_fct(x), "b.")
plt.plot(x, poisson.pair_correlation_function(x), "g")
plt.show()
