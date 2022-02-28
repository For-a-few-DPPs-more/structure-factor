import numpy as np
import pytest

import structure_factor.tapered_estimators as spe
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapered_estimators import allowed_k_scattering_intensity
from structure_factor.tapered_estimators import tapered_estimator_core as s_tp
from structure_factor.tapered_estimators import (
    tapered_estimator_debiased_direct as s_ddtp,
)
from structure_factor.tapered_estimators import (
    tapered_estimator_debiased_undirect as s_udtp,
)
from structure_factor.tapers import BartlettTaper, SineTaper

#! Please explain what is supposed to be tested, it is difficult to guess as is


class Taper1:
    @staticmethod
    def taper(x, window):
        return np.linalg.norm(x, axis=1)


class Taper2:
    @staticmethod
    def taper(x, window):
        n = x.shape[0]
        window_volume = window.volume
        return np.linalg.norm(x, axis=1) * np.sqrt(n / window_volume)


# test the results of the functions of tapered_estimators with simple tapers and trying to recover many dimension
#! unreadable test
@pytest.mark.parametrize(
    "k, points, window, taper, expected",
    [
        (
            np.array([[0, 0]]),
            np.array([[1, 3], [2, 0]]),
            BoxWindow([[-4, 4], [-6, 7]]),
            Taper1,
            2 + np.sqrt(10),
        ),
        (
            np.array([[0, 0, 0]]),
            np.array([[1, 3, 0], [2, 0, 1]]),
            BoxWindow([[-1, 2], [0, 3], [0, 1]]),
            Taper2,
            (np.sqrt(5) + np.sqrt(10)) * np.sqrt(2) / 3,
        ),
    ],
)
def test_tapered_dft(k, points, window, taper, expected):

    point_pattern = PointPattern(points, window)
    tapered_dft = spe.tapered_dft(k, point_pattern, taper)
    np.testing.assert_almost_equal(tapered_dft, expected)


@pytest.mark.parametrize(
    "dft, expected ",
    [
        (np.array([[1 + 2j], [2 - 1j], [3], [-2j]]), np.array([[5], [5], [9], [4]])),
    ],
)
def test_periodogram_from_dft(dft, expected):
    periodogram = spe.periodogram_from_dft(dft)
    np.testing.assert_almost_equal(periodogram, expected)


@pytest.mark.parametrize(
    "k, points, window, taper, expected ",
    [
        (
            np.array([[1, 2, 3]]),
            np.array(
                [[2 * np.pi, 2 * np.pi, 2 * np.pi], [4 * np.pi, 4 * np.pi, 4 * np.pi]]
            ),
            BoxWindow([[0, 10], [-2, 2], [3, 4]]),
            Taper2,
            108 * np.pi ** 2,
        )
    ],
)
def test_tapered_estimator_core(k, points, window, taper, expected):
    point_pattern = PointPattern(points, window)
    tp = spe.tapered_estimator_core(k, point_pattern, taper)
    np.testing.assert_almost_equal(tp, expected)


@pytest.mark.parametrize(
    " bounds, debiased_estimator",
    [
        ([[-1, 1], [-1, 1], [-1, 1]], s_ddtp),
        ([[-1, 1], [-1, 1]], s_ddtp),
        ([[-1, 1], [-1, 1], [-2, 1]], s_udtp),
        ([[-1, 1], [-1, 1]], s_udtp),
    ],
)
def test_debiased_and_non_debiased_estimators_are_equal_on_allowed_k_norm(
    bounds, debiased_estimator
):
    r"""Test that the debiased versions of :math:`\widehat{S}_{\mathrm{TP}}`
    gave the same results as :math:`\widehat{S}_{\mathrm{TP}}` on the allowed wavevector (i.e., where bias equal zero).
    """

    # creat pointpattern
    window = BoxWindow(bounds)
    points = window.rand(20)
    point_pattern = PointPattern(points, window)
    # creat allowed values
    L = np.diff(window.bounds)
    d = points.shape[1]
    k = allowed_k_scattering_intensity(d, L)

    taper = BartlettTaper
    s_debiased = debiased_estimator(k, point_pattern, taper)
    s_non_debiased = s_tp(k, point_pattern, taper)
    np.testing.assert_almost_equal(s_non_debiased, s_debiased)


@pytest.mark.parametrize(
    "name",
    ["undirect", "direct"],
)
def test_debiased_estimator_value_at_the_origin(name):
    r"""Test debiased versions of :math:`\widehat{S}_{\mathrm{TP}}` on simple case: :math:`k=0` and with the Bartlett taper."""
    N = 20
    window = BoxWindow([[0, 1], [0, 1], [0, 1]])
    points = window.rand(N)
    point_pattern = PointPattern(points, window)

    taper = BartlettTaper

    d = 3
    k = np.zeros((1, d))
    rho = point_pattern.intensity
    if name == "direct":
        s_estimated = s_ddtp(k, point_pattern, taper)
        s_expected = 1 / rho * (N - rho * taper.ft_taper(k, window)) ** 2
    if name == "undirect":
        s_estimated = s_udtp(k, point_pattern, taper)
        s_expected = (N ** 2) / rho - rho * taper.ft_taper(k, window) ** 2

    np.testing.assert_almost_equal(s_estimated, s_expected)


@pytest.mark.parametrize(
    "debiased, direct, monotaper",
    [[True, True, s_ddtp], [True, False, s_udtp], [False, False, s_tp]],
)
def test_multitapered_with_one_taper_equal_monotaper(debiased, direct, monotaper):
    r"""Test that multitapered estimator :math:`\widehat{S}_{\mathrm{MTP}}` and the corresponding debiased versions applied on only one taper, give the same results as :math:`\widehat{S}_{\mathrm{TP}}` and the corresponding debiased versions"""
    window = BoxWindow([[-5, 5], [-6, 5], [-5, 7]])
    points = window.rand(20)
    point_pattern = PointPattern(points, window)

    sf = StructureFactor(point_pattern)

    taper = SineTaper([1, 1, 2])
    tapers = [taper]
    k = np.random.rand(10, 3) * 6  # arbitrary points in 3-d

    k, s_estimated = sf.tapered_estimator(k, tapers, debiased=debiased, direct=direct)
    s_expected = monotaper(k, point_pattern, taper)
    np.testing.assert_array_equal(s_estimated, s_expected)
