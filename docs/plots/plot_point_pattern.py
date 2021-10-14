from structure_factor.point_pattern import PointPattern
from structure_factor.utils import get_random_number_generator

rng = get_random_number_generator(None)
points = rng.uniform(size=(100, 2))
pp = PointPattern(points)
pp.plot()
