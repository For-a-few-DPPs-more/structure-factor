from structure_factor.spatial_windows import BoxWindow
from structure_factor.hyperuniformity import subwindows

# Example 1: 
L = 30
window = BoxWindow([[-L / 2, L / 2]] * 2)
# subwindows and k
l_0 = 10
subwindows_box, k = subwindows(window, subwindows_type="BoxWindow", param_0=l_0)

# Example 2: 
subwindows_params=[1, 4, 7, 15]
subwindows_ball, k_norm = subwindows(window, subwindows_type="BallWindow", params=subwindows_params)

import matplotlib.pyplot as plt
fig, axis = plt.subplots(1, 2, figsize=(10,5))
axis[0].plot(0,0)
axis[1].plot(0,0)
for i, j in zip(subwindows_box, subwindows_ball):
    i.plot(axis=axis[0])
    j.plot(axis=axis[1])
plt.show()