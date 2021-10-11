from hypton.point_pattern import PointPattern
from hypton.utils import get_random_number_generator

rng = get_random_number_generator(None)
points = rng.uniform(size=(100, 2))
pp = PointPattern(points)
pp.plot()
