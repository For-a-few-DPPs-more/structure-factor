import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# creat box window
L = 70  # sidelength of the window
bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window

# restrict to box window
ginibre_pp_box = ginibre_pp.restrict_to_window(window)

# scattering intensity
sf_ginibre_box = StructureFactor(ginibre_pp_box)  # initialize the class StructureFactor
norm_k, si = sf_ginibre_box.scattering_intensity(k_max=6, meshgrid_shape=(200, 200))

# plot
sf_ginibre_box.plot_scattering_intensity(
    norm_k,
    si,
    plot_type="all",
    exact_sf=GinibrePointProcess.structure_factor,
    bins=60,  # number of bins
    error_bar=True,  # visualizing the error bars
)
