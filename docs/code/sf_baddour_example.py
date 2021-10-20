from structure_factor.data import load_data
from structure_factor.structure_factor import StructureFactor
import structure_factor.utils as utils
import numpy as np

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# initialize the class StructureFactor
sf_ginibre = StructureFactor(ginibre_pp)

# compute pair correlation function
pcf_fv = sf_ginibre.compute_pcf(method="fv", Kest=dict(rmax=45), 
                                fv=dict(method="b", spar=0.1))

# interpolate pcf_fv
domain, pcf_fv_func = sf_ginibre.interpolate_pcf(r=pcf_fv["r"], 
                                                 pcf_r=pcf_fv["pcf"],
                                                 clean=True)

# structure factor using Baddour Chouinard discrete hankel transform
rmax = domain["rmax"]
norm_k = np.linspace(0.3, 30, 2000)
norm_k, sf_BadChou = sf_ginibre.compute_sf_hankel_quadrature( pcf_fv_func, 
                                                        method="BaddourChouinard", 
                                                        norm_k=norm_k, 
                                                        rmax=rmax, 
                                                        nb_points=1000)

# plot
sf_ginibre.plot_sf_hankel_quadrature(norm_k,sf_BadChou,
                                    exact_sf=utils.structure_factor_ginibre,
                                    label="$S_{HBC}(k)$",
                                    error_bar=True,
                                    bins=100)
