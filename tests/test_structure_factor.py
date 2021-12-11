import numpy as np
import pytest

from structure_factor.data import load_data
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.utils import (
    pair_correlation_function_ginibre,
    structure_factor_ginibre,
)


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


def test_scattering_intensity(ginibre_pp):
    L = ginibre_pp.window.radius / np.sqrt(2)  # sidelength of the cubic window
    bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]
    window = BoxWindow(bounds)  # create a cubic window
    ginibre_pp_box = ginibre_pp.restrict_to_window(window)
    sf_pp = StructureFactor(ginibre_pp_box)
    k_norm, si = sf_pp.scattering_intensity(
        k_max=1,
        meshgrid_shape=(4, 4),
    )
    expected_k_norm = np.array(
        [
            1.38230077,
            1.04005223,
            1.01313319,
            1.38230077,
            1.04005223,
            0.50265482,
            0.44428829,
            1.04005223,
            1.01313319,
            0.44428829,
            0.37699112,
            1.01313319,
            1.38230077,
            1.04005223,
            1.01313319,
            1.38230077,
        ]
    )
    expected_si = np.array(
        [
            0.13647604,
            0.32944414,
            0.28438074,
            1.23781715,
            0.25049137,
            0.27434747,
            0.11169433,
            0.05213884,
            0.07547196,
            0.02700638,
            0.01302653,
            0.27565412,
            1.23781715,
            0.07421566,
            0.02024202,
            0.13647604,
        ]
    )
    np.testing.assert_almost_equal(k_norm, expected_k_norm)
    np.testing.assert_almost_equal(si, expected_si)


# def test_plot_scattering_intensity(ginibre_pp):
#     L = ginibre_pp.window.radius / np.sqrt(2)  # sidelength of the cubic window
#     bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]
#     window = BoxWindow(bounds)  # create a cubic window
#     ginibre_pp_box = ginibre_pp.restrict_to_window(window)
#     sf_pp = StructureFactor(ginibre_pp_box)
#     k_norm, si = sf_pp.scattering_intensity(
#         k_max=6,
#         meshgrid_shape=(50, 50),
#     )
#     sf_pp.plot_scattering_intensity(
#         k_norm,
#         si,
#         plot_type="all",
#         exact_sf=utils.structure_factor_ginibre,
#         bins=60,  # number of bins
#         error_bar=True,  # visualizing the error bars
#     )


def test_compute_structure_factor_ginibre_with_ogata(ginibre_pp):
    sf_pp = StructureFactor(ginibre_pp)
    method = "Ogata"
    params = dict(r_max=80, step_size=0.01, nb_points=1000)
    k_norm = np.linspace(1, 10, 1000)
    _, sf_computed = sf_pp.hankel_quadrature(
        pair_correlation_function_ginibre, k_norm=k_norm, method=method, **params
    )
    sf_expected = structure_factor_ginibre(k_norm)
    np.testing.assert_almost_equal(sf_computed, sf_expected)


def test_compute_structure_factor_ginibre_with_baddour_chouinard(ginibre_pp):
    sf_pp = StructureFactor(ginibre_pp)
    method = "BaddourChouinard"
    params = dict(r_max=80, nb_points=800)
    k_norm, sf_computed = sf_pp.hankel_quadrature(
        pair_correlation_function_ginibre, k_norm=None, method=method, **params
    )
    sf_expected = structure_factor_ginibre(k_norm)
    np.testing.assert_almost_equal(sf_computed, sf_expected)
    # fig = sf_pp.plot_sf_hankel_quadrature(
    #     k_norm, sf_computed, exact_sf=utils.structure_factor_ginibre
    # )
