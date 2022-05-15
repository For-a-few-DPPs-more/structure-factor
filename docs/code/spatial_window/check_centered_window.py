from structure_factor.spatial_windows import (
    BoxWindow,
    BallWindow,
    check_centered_window,
)

# Example 1:
window = BoxWindow(bounds=[[-1, 5], [-1, 2]])
check_centered_window(window)

# Example 2:
window = BallWindow(center=[0, 0], radius=5)
check_centered_window(window)
