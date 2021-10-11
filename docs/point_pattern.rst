Point pattern
##############

.. automodule:: hypton.point_pattern
    :members:
    :inherited-members:
    :show-inheritance:

.. code-block:: python


    from hypton.point_pattern import PointPattern
    from hypton.spatial_windows import BallWindow
    import numpy as np
    R = 100  # radius of the disk containing "points"
    center = [0, 0]  # center of the disk containing "points"
    window = BallWindow(center, R)  # creating ball window
    intensity=1 / np.pi # intensity of Ginibre Ensemble
    ginibre_pp = PointPattern(points, window, intensity)
    fig = ginibre_pp.plot()
