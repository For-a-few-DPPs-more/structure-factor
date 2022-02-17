# Generate a Ginibre PointPattern
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow

window = BallWindow(center=[0,0], radius=40) # Observation window
ginibre = GinibrePointProcess() # Ginibre process
ginibre_pp = ginibre.generate_point_pattern(window=window) # PointPattern
ginibre_pp.plot(file_name="ginibre_pp.pdf") # Plot and save the figure