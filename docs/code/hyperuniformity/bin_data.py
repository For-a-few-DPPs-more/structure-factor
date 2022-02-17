# Load Ginibre PointPattern and restrict to BoxWindow
from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow
from structure_factor.point_processes import GinibrePointProcess
import numpy as np

ginibre_pp = load_data.load_ginibre()
ginibre_pp_box = ginibre_pp.restrict_to_window(BoxWindow([[-35, 35], [-35, 35]]))

# Approximate the structure factor
from structure_factor.structure_factor import StructureFactor
sf_ginibre_box = StructureFactor(ginibre_pp_box)
## Wavevectors
x = np.linspace(0, 3, 80) 
x = x[x != 0] # Eliminate zero
X, Y = np.meshgrid(x, x)
k = np.column_stack((X.ravel(), Y.ravel())) # Wavevectors
s_ddmtp = sf_ginibre_box.multitapered_periodogram(k) # Estimate the structure factor

# Regularize the results
import structure_factor.utils as utils
from structure_factor.hyperuniformity import Hyperuniformity
k_norm = utils.norm_k(k)
hyperuniformity_test = Hyperuniformity(k_norm, s_ddmtp) # Initialize Hyperuniformity
k_norm_new, s_ddmtp_new, _ = hyperuniformity_test.bin_data(bins=40) # Regularization 

# Visualization of the results
import matplotlib.pyplot as plt
fig, axis = plt.subplots(figsize=(7,5))
axis.plot(k_norm, s_ddmtp, 'b,', label="Before regularization", rasterized=True)
axis.legend()
sf_ginibre_box.plot_isotropic_estimator(k_norm_new, s_ddmtp_new, axis=axis, color='m',
                                        exact_sf=GinibrePointProcess.structure_factor, 
                                        label="After regularization")
plt.show()