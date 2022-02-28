import matplotlib.pyplot as plt

import structure_factor.pair_correlation_function as pcf
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow

point_process = HomogeneousPoissonPointProcess(intensity=1)
window = BallWindow(center=[0, 0], radius=40)
point_pattern = point_process.generate_point_pattern(window=window)

pcf_estimated = pcf.estimate(
    point_pattern, method="fv", Kest=dict(rmax=20), fv=dict(method="c", spar=0.2)
)

pcf.plot(
    pcf_estimated,
    exact_pcf=point_process.pair_correlation_function,
    figsize=(7, 6),
    color=["grey"],
    style=["."],
)
plt.tight_layout(pad=1)
