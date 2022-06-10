import numpy as np
from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow
from structure_factor.hyperuniformity import subwindows, multiscale_test

# PointPattern list
poisson = HomogeneousPoissonPointProcess(intensity=1 / np.pi)
N = 500
L = 160
window = BoxWindow([[-L / 2, L / 2]] * 2)
point_patterns = [
    poisson.generate_point_pattern(window=window) for _ in range(N)
]
# subwindows and k
l_0 = 40
subwindows_list, k = subwindows(window, subwindows_type="BoxWindow", param_0=l_0)

# multiscale hyperuniformity test
mean_poisson = 90
summary = multiscale_test(point_patterns,
    estimator="scattering_intensity", k_list=k,
    subwindows_list=subwindows_list, mean_poisson=mean_poisson,
    verbose=True)

import matplotlib.pyplot as plt
plt.hist(summary["Z"], color="grey", label="Z")
plt.axvline(summary["mean_Z"], color="b", linewidth=1, label=r"$\bar{Z}$")
plt.axvline(summary["mean_Z"] - 3*summary["std_mean_Z"], linestyle='dashed',
            color="k", linewidth=1, label=r"$\bar{Z} \pm 3 \bar{\sigma}/ \sqrt{N}$")
plt.axvline(summary["mean_Z"] + 3*summary["std_mean_Z"], linestyle='dashed',
            color="k", linewidth=1)
plt.xlabel(r"$Z$")
plt.title(r"Histogram of the $N$ obtained values of $Z$")
plt.legend()
plt.show()