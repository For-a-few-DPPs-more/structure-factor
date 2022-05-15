from structure_factor.spatial_windows import BoxWindow, check_cubic_window

# Example 1:
window = BoxWindow(bounds=[[-1, 5], [-1, 2]])
check_cubic_window(window)

# Example 2:
window = BoxWindow(bounds=[[-1, 2], [-1, 2], [-1, 2]])
check_cubic_window(window)
