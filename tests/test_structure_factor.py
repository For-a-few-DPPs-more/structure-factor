import os
import pickle

import hypton
import numpy as np
import numpy.random as npr
import pytest
from hypton.structure_factor import StructureFactor

direc = os.path.dirname(os.path.abspath(__file__))
my_data_path = os.path.join(direc, os.pardir, "data/test_pp.pickle")
with open(my_data_path, "rb") as handle:
    ginibre_pp = pickle.load(handle)


def pcf_ginibre(x):
    return 1 - np.exp(-(x ** 2))


def sf_ginibre(x):
    return 1 - np.exp(-(x ** 2) / 4)


r = np.linspace(0, 80, 500)
k = np.linspace(1, 10, 1000)


@pytest.mark.parametrize("r, pcf", [(r, pcf_ginibre(r))])
def test_interpolate_pcf(r, pcf):
    sf_pp = StructureFactor(ginibre_pp)
    _, result_pcf = sf_pp.interpolate_pcf(r, pcf)
    x = np.linspace(5, 10, 30)
    np.testing.assert_almost_equal(pcf_ginibre(x), result_pcf(x))


@pytest.mark.parametrize("pp, pcf, norm_k, rmax", [(ginibre_pp, pcf_ginibre, k, 80)])
def test_compute_sf_hankel_quadrature_Ogata(pp, pcf, norm_k, rmax):
    sf_pp = StructureFactor(pp)
    norm_k, sf = sf_pp.compute_sf_hankel_quadrature(
        pcf, norm_k=norm_k, rmax=rmax, step_size=0.01, nb_points=1000
    )
    np.testing.assert_almost_equal(sf_ginibre(norm_k), sf)


@pytest.mark.parametrize("pp, pcf, rmax", [(ginibre_pp, pcf_ginibre, 80)])
def test_compute_sf_hankel_quadrature_Baddour(pp, pcf, rmax):
    sf_pp = StructureFactor(pp)
    norm_k, sf = sf_pp.compute_sf_hankel_quadrature(
        pcf, method="BaddourChouinard", rmax=rmax, nb_points=800
    )
    np.testing.assert_almost_equal(sf_ginibre(norm_k), sf)
