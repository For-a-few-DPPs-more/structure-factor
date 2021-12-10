import numpy as np
import pytest
from structure_factor.point_pattern import PointPattern
import structure_factor.spectral_estimator as spe
from structure_factor.spatial_windows import BoxWindow


def taper_1(x, window):
    return np.linalg.norm(x)


def taper_2(x, window):
    n = x.shape[0]
    window_volume = window.volume
    return np.linalg.norm(x, axis=1) * np.sqrt(n / window_volume)


@pytest.mark.parametrize(
    "k, points, taper, expected",
    [
        (2 * np.pi * 7, np.random.randn(20, 1), 0, 0),
        (np.array([[0, 0, 0, 0]]), np.random.randn(10, 4), 1 / 20, 10 / 20),
        (
            np.array([[0, 0]]),
            np.array([[1, 3], [2, 0], [3, 7]]),
            taper_1,
            3 * taper_1(np.array([[1, 3], [2, 0], [3, 7]]), 1),
        ),
        (
            2 * np.pi * np.array([[1 / 2, 1 / 2, 1 / 2], [1, 1, 1]]),
            np.array([[1, 1, 1], [3, 3, 3], [7, 7, 7]]),
            1 / 10,
            np.array([-3 / 10, 3 / 10]),
        ),
    ],
)
def test_tapered_dft(k, points, taper, expected):
    point_pattern = PointPattern(points)
    tapered_dft = spe.tapered_dft(k, point_pattern, taper)
    np.testing.assert_almost_equal(tapered_dft, expected)


@pytest.mark.parametrize(
    "k, points, window, taper, expected ",
    [
        (
            np.array([[1, 2, 3]]),
            np.array(
                [[2 * np.pi, 2 * np.pi, 2 * np.pi], [4 * np.pi, 4 * np.pi, 4 * np.pi]]
            ),
            BoxWindow([[0, 10], [-2, 2], [3, 4]]),
            taper_2(
                np.array(
                    [
                        [2 * np.pi, 2 * np.pi, 2 * np.pi],
                        [4 * np.pi, 4 * np.pi, 4 * np.pi],
                    ]
                ),
                BoxWindow([[0, 10], [-2, 2], [3, 4]]),
            ),
            108 * np.pi ** 2,
        )
    ],
)
def test_tapered_periodogram(k, points, window, taper, expected):
    point_pattern = PointPattern(points, window)
    tp = spe.tapered_periodogram(k, point_pattern, taper)
    np.testing.assert_almost_equal(tp, expected)
