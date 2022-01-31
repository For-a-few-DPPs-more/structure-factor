import numpy as np
import pytest

import structure_factor.spectral_estimators as spe
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow

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
