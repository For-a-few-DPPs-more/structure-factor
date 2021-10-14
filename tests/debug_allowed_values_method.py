import os

import numpy as np
from hypton import utils
from hypton.point_pattern import PointPattern
import pickle
from hypton.structure_factor import StructureFactor


def pcf_ginibre(x):
    return 1 - np.exp(-(x ** 2))


def sf_ginibre(x):
    return 1 - np.exp(-(x ** 2) / 4)


r = np.linspace(0, 80, 500)
k = np.linspace(1, 10, 1000)

direc = os.path.dirname(os.path.abspath(__file__))
my_data_path = os.path.join(direc, os.pardir, "data/test_pp.pickle")
with open(my_data_path, "rb") as handle:
    ginibre_pp = pickle.load(handle)

sf_pp = StructureFactor(ginibre_pp)
norm_k, sf = sf_pp.compute_sf_hankel_quadrature(
    pcf_ginibre, norm_k=k, rmax=80, step_size=0.01, nb_points=1000
)
print(sf == sf_ginibre(norm_k))
print(ginibre_pp.points.shape)