import numpy as np
import pytest
import structure_factor.utils as utils
import scipy.special as sc

from structure_factor.data import load_data
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow, BallWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.utils import (
    pair_correlation_function_ginibre,
    structure_factor_ginibre,
)
from structure_factor.tapers import BartlettTaper


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


# todo reshape the test, scattering_intensity does not return k_norm but k
#
def test_scattering_intensity(ginibre_pp):
    """Test scattering intensity of the Ginibre on allowed wavevectors"""
    L = ginibre_pp.window.radius / np.sqrt(2)  # sidelength of the cubic window
    bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]
    window = BoxWindow(bounds)  # create a cubic window
    ginibre_pp_box = ginibre_pp.restrict_to_window(window)
    sf_pp = StructureFactor(ginibre_pp_box)
    k, si = sf_pp.scattering_intensity(
        k_max=1,
        meshgrid_shape=(2, 3),
    )
    k_norm = utils.norm_k(k)
    expected_k_norm = np.array(
        [
            1.38230077,
            1.38230077,
            1.04005223,
            1.04005223,
            1.01313319,
            1.01313319,
            1.38230077,
            1.38230077,
        ]
    )
    expected_si = np.array(
        [
            0.13642892,
            1.23738984,
            0.2504049,
            0.05212084,
            0.07544591,
            0.27555896,
            1.23738984,
            0.13642892,
        ]
    )
    np.testing.assert_almost_equal(k_norm, expected_k_norm)
    np.testing.assert_almost_equal(si, expected_si)


@pytest.mark.parametrize(
    "k, points, unscaled_expected",
    [
        (np.array([[1], [2]]), np.array([[np.pi], [2 * np.pi]]), np.array([0, 4])),
    ],
)
def test_scattering_intensity2(k, points, unscaled_expected):
    """Test the scattering intensity and the debiased versions on simple choice of wavevectors in 1-d, and specific points allowing to simplify the calculation"""
    window = BoxWindow([-8, 8])
    point_pattern = PointPattern(points, window)
    sf = StructureFactor(point_pattern)
    intensity = point_pattern.intensity
    _, si = sf.scattering_intensity(k, debiased=False)
    expected_si = unscaled_expected / (window.volume * intensity)

    _, si_ddtp = sf.scattering_intensity(k, debiased=True, direct=True)
    expected_si_ddtp = (
        1
        / intensity
        * (
            np.sqrt(unscaled_expected / window.volume)
            - intensity * BartlettTaper.ft_taper(k, window)
        )
        ** 2
    )

    _, si_udtp = sf.scattering_intensity(k, debiased=True, direct=False)
    expected_si_udtp = expected_si - intensity * BartlettTaper.ft_taper(k, window) ** 2
    np.testing.assert_almost_equal(si, expected_si)
    np.testing.assert_almost_equal(si_ddtp, expected_si_ddtp)
    np.testing.assert_almost_equal(si_udtp, expected_si_udtp)


def test_tapered_periodogram():
    r"""Test that the debiased versions of :math:`\widehat{S}_{\mathrm{TP}}` with Bartlett taper gave the scattering intensity and the debiased verions."""
    points = np.random.rand(20, 3) * 5
    window = BoxWindow([[-6, 6], [-7, 5], [-5, 6]])
    taper = BartlettTaper

    point_pattern = PointPattern(points, window)
    sf = StructureFactor(point_pattern)
    k = np.random.randn(10, 3)

    # scattering intensity on k
    _, si_k = sf.scattering_intensity(k, debiased=False)
    s_tp = sf.tapered_periodogram(k, taper, debiased=False)
    # directly debiased scattering intensity
    _, si_dd = sf.scattering_intensity(k, debiased=True, direct=True)
    s_ddtp = sf.tapered_periodogram(k, taper, debiased=True, direct=True)
    # undirectly debiased scattering intensity
    _, si_ud = sf.scattering_intensity(k, debiased=True, direct=False)
    s_udtp = sf.tapered_periodogram(k, taper, debiased=True, direct=False)

    np.testing.assert_almost_equal(si_k, s_tp)
    np.testing.assert_almost_equal(si_dd, s_ddtp)
    np.testing.assert_almost_equal(si_ud, s_udtp)


def test_bartlett_isotropic_estimator():
    r"""Test the estimator :math:`\widehat{S}_{\mathrm{BI}}`, for simple choice of parameters in 3-d"""
    points = np.random.rand(2, 3) * 3
    d = points.shape[1]
    r = 4
    window = BallWindow(center=[0, 0, 0], radius=r)
    window_volume = 4 / 3 * np.pi * r ** 3
    window_surface = 4 * np.pi
    point_pattern = PointPattern(points, window)
    k_norm = np.array([1.0])
    # s_bi
    sf = StructureFactor(point_pattern)
    _, s_bi = sf.bartlett_isotropic_estimator(k_norm)
    # expected s_bi
    dist_points = np.linalg.norm(points[0] - points[1])
    m = (2 * np.pi) ** (d / 2) / (
        point_pattern.intensity * window_volume * window_surface
    )
    expected_s_bi = 1 + m * 2 * sc.jv(d / 2 - 1, dist_points) / (
        dist_points ** (d / 2 - 1)
    )
    np.testing.assert_almost_equal(s_bi, expected_s_bi)


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
