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
poisson_pcf_fct = pcf.interpolate(r=r, pcf_r=pcf_r, drop=True)

# Initialize the class StructureFactor
from structure_factor.structure_factor import StructureFactor

sf_poisson = StructureFactor(poisson_pp)

# Estimating the structure factor using Baddour Chouinard quadrature
k_norm = np.linspace(1, 10, 500)  # vector of wave length
k_norm, s_hbc = sf_poisson.hankel_quadrature(
    poisson_pcf_fct, method="BaddourChouinard", k_norm=k_norm, r_max=20, nb_points=1000
)
fig, axis = plt.subplots(figsize=(7, 6))
fig = sf_poisson.plot_isotropic_estimator(
    k_norm,
    s_hbc,
    axis=axis,
    error_bar=True,
    bins=30,
    label=r"$\widehat{S}_{\mathrm{BC}}(k)$",
)
