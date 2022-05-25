from structure_factor.spatial_windows import (
    BoxWindow,
    BallWindow,
    subwindow_parameter_max,
)

# Example 1:
window1 = BoxWindow([[-8, 8], [-7, 7]])
r = subwindow_parameter_max(window=window1, subwindow_type="BallWindow")
subwindow1 = BallWindow(center=[0,0], radius=r)
# Example 2:
window2 = BallWindow(center=[0, 0], radius=5)
l = subwindow_parameter_max(window=window2)
subwindow2 = BoxWindow([[-l/2,l/2]]*2)

import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 2, figsize=(10,5))
axis[0].plot(0,0)
window1.plot(axis=axis[0], color="b", label="Window")
subwindow1.plot(axis=axis[0], color="k", label="Subwindow")
axis[0].legend()
axis[0].title.set_text("Example 1")

axis[1].plot(0,0)
window2.plot(axis=axis[1], color="b", label="Window")
subwindow2.plot(axis=axis[1], color="k", label="Subwindow")
axis[1].legend()
axis[1].title.set_text("Example 2")

plt.show()