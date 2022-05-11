from structure_factor.data import load_data
from structure_factor.multiscale_estimators import subwindows_list, multiscale_estimator


# PointPattern 
point_pattern = load_data.load_ginibre()
window = point_pattern.window

# subwindows and k
l_0 = 40
subwindows_list, k = subwindows_list(window, subwindows_type="BoxWindow", param_0=l_0)

# multiscale_estimator
mean_poisson = 85
z = multiscale_estimator(point_pattern, estimator="scattering_intensity", 
                               k_list=k, subwindows_list=subwindows_list, 
                               mean_poisson=mean_poisson)

z