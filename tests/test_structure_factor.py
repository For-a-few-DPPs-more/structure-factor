import numpy as np
import pytest
import scipy.special as sc

import structure_factor.utils as utils
from structure_factor.data import load_data
from structure_factor.point_pattern import PointPattern
from structure_factor.point_processes import GinibrePointProcess
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapers import BartlettTaper


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


# todo reshape the test, scattering_intensity does not return k_norm but k
@pytest.mark.parametrize(
    "to_test",
    ["k_norm", "s_si"],
)
def test_scattering_intensity_of_ginibre_on_allowed_k_norm(ginibre_pp, to_test):
    """Test scattering intensity of the Ginibre on allowed wavevectors"""
    # PointPattern
    point_pattern = ginibre_pp

    # Restric to BoxWindpw
    L = point_pattern.window.radius / np.sqrt(2)  # sidelength of the cubic window
    bounds = [[-L / 2, L / 2], [-L / 2, L / 2]]
    window = BoxWindow(bounds)  # create a cubic window
    ginibre_pp_box = point_pattern.restrict_to_window(window)

    # Initialize StructureFactor
    sf_pp = StructureFactor(ginibre_pp_box)
    k, s_si = sf_pp.scattering_intensity(
        k_max=1,
        meshgrid_shape=(2, 3),
    )
    if to_test == "k_norm":
        expected = utils.norm(k)
        tested = np.array(
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
    elif to_test == "s_si":
        tested = s_si
        expected = np.array(
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
    np.testing.assert_almost_equal(tested, expected)


@pytest.mark.parametrize(
    "debiased, direct",
    [(False, False), (True, True), (True, False)],
)
def test_scattering_intensity_on_specific_points_and_wavevector(debiased, direct):
    """Test the scattering intensity and the debiased versions on simple choice of wavevectors in 1-d, and specific points allowing to simplify the calculation"""
    # BoxWindow
    L = 8
    window = BoxWindow([-L, L])

    # PointPattern
    points = np.array([[np.pi], [2 * np.pi]])
    point_pattern = PointPattern(points, window)

    # StructureFactor
    sf = StructureFactor(point_pattern)
    rho = point_pattern.intensity

    # Scattering intensity
    k = np.array([[1], [2]])
    _, result = sf.scattering_intensity(k, debiased=debiased, direct=direct)

    taper = BartlettTaper()
    dft = np.sum(np.exp(-1j * k.dot(points.T)), axis=1)
    dft /= np.sqrt(window.volume)

    if not direct and not debiased:
        expected = np.abs(dft) ** 2 / rho
    elif not direct and debiased:
        expected = np.abs(dft) ** 2 / rho - rho * np.abs(taper.ft_taper(k, window)) ** 2
    elif direct and debiased:
        expected = (np.abs(dft - rho * taper.ft_taper(k, window)) ** 2) / rho

    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize(
    "debiased, direct",
    [(False, False), (True, True), (True, False)],
)
def test_tapered_estimator_with_bartlett_taper_equal_scattering_intensity(
    debiased, direct
):
    r"""Test that the debiased versions of :math:`\widehat{S}_{\mathrm{TP}}` with Bartlett taper gave the scattering intensity and the debiased verions."""

    # PointPattern
    window = BoxWindow([[-6, 6], [-7, 5], [-5, 6]])
    points = window.rand(20)
    point_pattern = PointPattern(points, window)

    # StructureFactor
    sf = StructureFactor(point_pattern)
    k = np.random.randn(10, 3)
    tapers = [BartlettTaper()]

    # Scattering intensity and tapered estimator
    _, s_si = sf.scattering_intensity(k, debiased=debiased, direct=direct)
    k, s_tp = sf.tapered_estimator(k, tapers=tapers, debiased=debiased, direct=direct)

    np.testing.assert_almost_equal(s_si, s_tp)


def test_tapered_estimators_isotropic_on_origin():
    r"""Test the estimator :math:`\widehat{S}_{\mathrm{BI}}`, for simple choice of parameters in 3-d"""
    # window
    r = 4
    window = BallWindow(center=[0, 0, 0], radius=r)
    points = window.rand(2)

    # PointPattern
    point_pattern = PointPattern(points, window)
    k_norm = np.array([1.0])

    # s_bi
    sf = StructureFactor(point_pattern)
    _, s_bi = sf.bartlett_isotropic_estimator(k_norm)

    # Expected s_bi
    d = points.shape[1]
    window_volume = 4 / 3 * np.pi * r ** 3
    window_surface = 4 * np.pi
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
    _, sf_computed = sf_pp.quadrature_estimator_isotropic(
        GinibrePointProcess.pair_correlation_function,
        k_norm=k_norm,
        method=method,
        **params
    )
    sf_expected = GinibrePointProcess.structure_factor(k_norm)
    np.testing.assert_almost_equal(sf_computed, sf_expected)


def test_compute_structure_factor_ginibre_with_baddour_chouinard(ginibre_pp):
    sf_pp = StructureFactor(ginibre_pp)
    method = "BaddourChouinard"
    params = dict(r_max=80, nb_points=800)
    k_norm, sf_computed = sf_pp.quadrature_estimator_isotropic(
        GinibrePointProcess.pair_correlation_function,
        k_norm=None,
        method=method,
        **params
    )
    sf_expected = GinibrePointProcess.structure_factor(k_norm)
    np.testing.assert_almost_equal(sf_computed, sf_expected)
