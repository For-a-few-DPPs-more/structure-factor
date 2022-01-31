import numpy as np

import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.structure_factor import StructureFactor

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# initialize the class StructureFactor
sf_ginibre = StructureFactor(ginibre_pp)

# approximate the pair correlation function
pcf_fv = sf_ginibre.compute_pcf(
    method="fv", Kest=dict(r_max=45), fv=dict(method="b", spar=0.1)
)

# interpolate pcf_fv["pcf"]
domain, pcf_fv_func = sf_ginibre.interpolate_pcf(
    r=pcf_fv["r"], pcf_r=pcf_fv["pcf"], clean=True
)

# structure factor using Baddour Chouinard discrete Hankel transform
r_max = domain["r_max"]
k_norm = np.linspace(0.3, 30, 2000)
k_norm, sf_BadChou = sf_ginibre.hankel_quadrature(
    pcf_fv_func, k_norm=k_norm, method="BaddourChouinard", r_max=r_max, nb_points=1000
)

# plot
sf_ginibre.plot_sf_hankel_quadrature(
    k_norm,
    sf_BadChou,
    exact_sf=GinibrePointProcess.structure_factor,
    label=r"$S_{HBC}(k)$",
    error_bar=True,
    bins=100,
)
