import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.structure_factor import StructureFactor

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# initialize the class StructureFactor
sf_ginibre = StructureFactor(ginibre_pp)

# compute the pair correlation function
pcf_fv = sf_ginibre.compute_pcf(
    method="fv", Kest=dict(rmax=45), fv=dict(method="b", spar=0.1)
)

# plot
sf_ginibre.plot_pcf(
    pcf_fv,
    exact_pcf=utils.pair_correlation_function_ginibre,
    figsize=(10, 6),
    color=["grey", "b", "darkcyan"],
    style=[".", "o", "^"],
)
