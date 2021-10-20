from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow

# load Ginibre PointPattern
ginibre_pp = load_data.load_ginibre()

# creat box window
L = 70  # sidelength of the window
bounds = np.array([[-L / 2, L / 2], [-L / 2, L / 2]])  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window

# restrict to window
ginibre_pp_box = ginibre_pp.restrict_to_window(window)

# plot the result
ginibre_pp_box.plot()
