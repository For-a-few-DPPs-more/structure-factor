import numpy as np
import pytest

import structure_factor.spectral_estimators as spe
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow

#! Please explain what is supposed to be tested, it is difficult to guess as is


class Taper1:
    @staticmethod
    def taper(x, window):
        return np.linalg.norm(x)


class Taper2:
    @staticmethod
    def taper(x, window):
        n = x.shape[0]
        window_volume = window.volume
        return np.linalg.norm(x, axis=1) * np.sqrt(n / window_volume)


#! unreadable test
# @pytest.mark.parametrize(
#     "k, points, taper, expected",
#     [
#         (2 * np.pi * 7, np.random.randn(20, 1), 0, 0),
#         (np.array([[0, 0, 0, 0]]), np.random.randn(10, 4), 1 / 20, 10 / 20),
#         (
#             np.array([[0, 0]]),
#             np.array([[1, 3], [2, 0], [3, 7]]),
#             Taper1,
#             3 * Taper1(np.array([[1, 3], [2, 0], [3, 7]]), 1),
#         ),
#         (
#             2 * np.pi * np.array([[1 / 2, 1 / 2, 1 / 2], [1, 1, 1]]),
#             np.array([[1, 1, 1], [3, 3, 3], [7, 7, 7]]),
#             1 / 10,
#             np.array([-3 / 10, 3 / 10]),
#         ),
#     ],
# )
# def test_tapered_dft(k, points, taper, expected):
#     point_pattern = PointPattern(points)
#     tapered_dft = spe.tapered_dft(k, point_pattern, taper)
#     np.testing.assert_almost_equal(tapered_dft, expected)


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
def test_tapered_periodogram(k, points, window, taper, expected):
    point_pattern = PointPattern(points, window)
    tp = spe.tapered_spectral_estimator_core(k, point_pattern, taper)
    np.testing.assert_almost_equal(tp, expected)
