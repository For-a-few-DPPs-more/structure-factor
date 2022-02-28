import matplotlib.pyplot as plt
import numpy as np

import structure_factor.pair_correlation_function as pcf
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BallWindow
from structure_factor.structure_factor import StructureFactor

point_process = HomogeneousPoissonPointProcess(intensity=1)

window = BallWindow(center=[0, 0], radius=40)
point_pattern = point_process.generate_point_pattern(window=window)

pcf_estimated = pcf.estimate(
    point_pattern, method="fv", Kest=dict(rmax=20), fv=dict(method="c", spar=0.2)
)

# Interpolate/extrapolate the results
r = pcf_estimated["r"]
pcf_r = pcf_estimated["pcf"]
pcf_interpolated = pcf.interpolate(r=r, pcf_r=pcf_r, drop=True)

# Estimate the structure factor using Baddour Chouinard quadrature
sf = StructureFactor(point_pattern)
k_norm = np.linspace(1, 10, 500)  # vector of wavelength
k_norm, sf_estimated = sf.quadrature_estimator_isotropic(
    pcf_interpolated, method="BaddourChouinard", k_norm=k_norm, r_max=20, nb_points=1000
)

fig, ax = plt.subplots(figsize=(7, 6))
fig = sf.plot_isotropic_estimator(
    k_norm,
    sf_estimated,
    axis=ax,
    error_bar=True,
    bins=30,
    label=r"$\widehat{S}_{\mathrm{BC}}(k)$",
)
plt.tight_layout(pad=1)
