from structure_factor.spatial_windows import BoxWindow, BallWindow
from structure_factor.multiscale_estimators import m_threshold

window_min = BallWindow(center=[0,0], radius=4)
window_max = BoxWindow(bounds=[[-10, 10], [-10, 10]])
m_thresh = m_threshold(window_min, window_max)

import matplotlib.pyplot as plt
fig, axis = plt.subplots( figsize=(5,5))
axis.plot(0,0)
window_min.plot(axis=axis, color="k", label="smallest window")
window_max.plot(axis=axis, color="b", label="biggest window")
for j in range(1, m_thresh+1):
    w = BallWindow(center=[0,0], radius=4+j)
    w.plot(axis=axis, color="grey")
axis.legend()
plt.show()