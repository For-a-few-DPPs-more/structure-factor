from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0, 0], radius=4)
print("The volume of the window is equal to", window.volume)

from structure_factor.spatial_windows import BoxWindow

bounds = np.array([[-2, 2], [-2, 2], [-2, 2]])
window = BoxWindow(bounds)
print("The volume of the window is equal to", window.volume)
