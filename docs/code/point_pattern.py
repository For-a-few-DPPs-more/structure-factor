from structure_factor.data import load_data  # import data

# load points
points = load_data.load_ginibre().points

from structure_factor.point_pattern import PointPattern

# Example with Ballwindow
from structure_factor.spatial_windows import BallWindow
# create BallWindow
window = BallWindow(center=[0, 0], radius=100)
# create PointPattern object
point_pattern = PointPattern(points, window)

# Example with Boxwindow
from structure_factor.spatial_windows import BoxWindow
# create BoxWindow
L = 70  # sidelength of the window
bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]  # bounds of the window
window = BoxWindow(bounds)  # create a cubic window
# create PointPattern object
point_pattern = PointPattern(points, window)
