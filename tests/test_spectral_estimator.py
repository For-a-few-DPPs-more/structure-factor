from ctypes import util
import numpy as np
import pytest

import structure_factor.spectral_estimators as spe
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow
import structure_factor.utils as utils
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


# test the results of the functions of spectral_estimators with simple tapers and trying to recover many dimension
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
def test_tapered_spectral_estimator_core(k, points, window, taper, expected):
    point_pattern = PointPattern(points, window)
    tp = spe.tapered_spectral_estimator_core(k, point_pattern, taper)
    np.testing.assert_almost_equal(tp, expected)


@pytest.mark.parametrize(
    "points, L, window, taper",
    [
        (
            np.random.rand(10, 3),  # in 3-d random points
            2,
            BoxWindow([[-1, 1], [-1, 1], [-1, 1]]),
            BartlettTaper,
        ),
        (
            np.random.rand(10, 2),  # in 2-d random points
            2,
            BoxWindow([[-1, 1], [-1, 1]]),
            BartlettTaper,
        ),
    ],
)
def test_tapered_spectral_estimator_debiased(points, L, window, taper):
    r"""Test that the debiased versions of :math:`\widehat{S}_{\mathrm{TP}}`
    gave the same results as :math:`\widehat{S}_{\mathrm{TP}}` on the allowed wavevector (i.e., where bias equal zero)
    """
    d = points.shape[1]
    point_pattern = PointPattern(points, window)
    k = utils.allowed_wave_vectors(d, L)
    s_ddtp = spe.tapered_spectral_estimator_debiased_direct(k, point_pattern, taper)
    s_udtp = spe.tapered_spectral_estimator_debiased_undirect(k, point_pattern, taper)
    s_tp = spe.tapered_spectral_estimator_core(k, point_pattern, taper)
    np.testing.assert_almost_equal(s_tp, s_ddtp)
    np.testing.assert_almost_equal(s_tp, s_udtp)


def test_tapered_spectral_estimator_debiased2():
    r"""Test debiased versions of :math:`\widehat{S}_{\mathrm{TP}}` on simple case: :math:`k=0` and with the Bartlett taper."""
    k = np.array([[0, 0, 0]])
    points = np.random.rand(20, 3)
    window = BoxWindow([[0, 1], [0, 1], [0, 1]])
    point_pattern = PointPattern(points, window)
    taper = BartlettTaper
    s_ddtp = spe.tapered_spectral_estimator_debiased_direct(k, point_pattern, taper)
    s_udtp = spe.tapered_spectral_estimator_debiased_undirect(k, point_pattern, taper)
    intensity = point_pattern.intensity
    expected_s_ddtp = (
        1 / intensity * (20 - intensity * BartlettTaper.ft_taper(k, window)) ** 2
    )
    expected_s_udtp = (20 ** 2) / intensity - intensity * BartlettTaper.ft_taper(
        k, window
    ) ** 2
    np.testing.assert_almost_equal(expected_s_ddtp, s_ddtp)
    np.testing.assert_almost_equal(expected_s_udtp, s_udtp)


def test_multitapered_spectral_estimator():
    r"""Test that multitapered estimator :math:`\widehat{S}_{\mathrm{MTP}}` and the corresponding debiased versions applied on only one taper, give the same results as :math:`\widehat{S}_{\mathrm{TP}}` and the corresponding debiased versions"""
    points = np.random.rand(20, 3) * 10  # arbitrary points in 3-d
    window = BoxWindow([[-5, 5], [-6, 5], [-5, 7]])
    point_pattern = PointPattern(points, window)
    k = np.random.rand(10, 3) * 6  # arbitrary points in 3-d
    p = [1, 1, 2]
    taper = SineTaper(p)
    tapers = [taper]
    s_mtp = spe.multitapered_spectral_estimator(
        k, point_pattern, *tapers, debiased=False
    )
    s_mddtp = spe.multitapered_spectral_estimator(
        k, point_pattern, *tapers, debiased=True, direct=True
    )
    s_mudtp = spe.multitapered_spectral_estimator(
        k, point_pattern, *tapers, debiased=True, direct=False
    )
    s_tp = spe.tapered_spectral_estimator_core(k, point_pattern, taper)
    s_ddtp = spe.tapered_spectral_estimator_debiased_direct(k, point_pattern, taper)
    s_udtp = spe.tapered_spectral_estimator_debiased_undirect(k, point_pattern, taper)
    np.testing.assert_almost_equal(s_mtp, s_tp)
    np.testing.assert_almost_equal(s_mddtp, s_ddtp)
    np.testing.assert_almost_equal(s_mudtp, s_udtp)
