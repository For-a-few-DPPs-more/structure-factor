import matplotlib.pyplot as plt
import numpy as np

import structure_factor.pair_correlation_function as pcf
from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow

point_process = HomogeneousPoissonPointProcess(intensity=1)
window = BallWindow(center=[0, 0], radius=40)
point_pattern = point_process.generate_point_pattern(window=window)

pcf_estimated = pcf.estimate(
    point_pattern, method="fv", Kest=dict(rmax=20), fv=dict(method="c", spar=0.2)
)

# Interpolate/extrapolate the results
r = pcf_estimated["r"]
pcf_r = pcf_estimated["pcf"]
pcf_interpolated = pcf.interpolate(
    r=r,
    pcf_r=pcf_r,
    drop=True,
)

x = np.linspace(0, 50, 200)
plt.plot(x, pcf_interpolated(x), "b.")
plt.plot(x, point_process.pair_correlation_function(x), "g")
plt.tight_layout(pad=1)
