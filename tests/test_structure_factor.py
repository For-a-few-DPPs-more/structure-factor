import numpy as np
import pytest

from hypton.data import load_data
from hypton.structure_factor import StructureFactor
from hypton.utils import pair_correlation_function_ginibre, structure_factor_ginibre


@pytest.fixture
def radius():
    return np.linspace(0, 80, 500)


@pytest.fixture
def norm_k():
    return np.linspace(1, 10, 1000)


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


def test_interpolate_pcf_ginibre(ginibre_pp, radius):
    sf_pp = StructureFactor(ginibre_pp)
    pcf_r = pair_correlation_function_ginibre(radius)
    _, interp_pcf = sf_pp.interpolate_pcf(radius, pcf_r)
    x = np.linspace(5, 10, 30)
    computed_pcf = interp_pcf(x)
    expected_pcf = pair_correlation_function_ginibre(x)
    np.testing.assert_almost_equal(computed_pcf, expected_pcf)


def test_compute_structure_factor_ginibre_with_ogata(ginibre_pp, norm_k):
    sf_pp = StructureFactor(ginibre_pp)
    method = "Ogata"
    params = dict(rmax=80, step_size=0.01, nb_points=1000)
    norm_k, sf_computed = sf_pp.compute_sf_hankel_quadrature(
        pair_correlation_function_ginibre, norm_k=norm_k, method=method, **params
    )
    sf_expected = structure_factor_ginibre(norm_k)
    np.testing.assert_almost_equal(sf_computed, sf_expected)


def test_compute_structure_factor_ginibre_with_baddour_chouinard(ginibre_pp):
    sf_pp = StructureFactor(ginibre_pp)
    method = "BaddourChouinard"
    params = dict(rmax=80, nb_points=800)
    norm_k, sf_computed = sf_pp.compute_sf_hankel_quadrature(
        pair_correlation_function_ginibre, norm_k=None, method=method, **params
    )
    sf_expected = structure_factor_ginibre(norm_k)
    np.testing.assert_almost_equal(sf_computed, sf_expected)
