import numpy as np
import pytest

import structure_factor.spectral_estimator as spe


def taper_function(x):
    return np.linalg.norm(x)


@pytest.mark.parametrize(
    "k, points, taper, expected",
    [
        (2 * np.pi * 7, np.random.randn(20, 1), 0, 0),
        (np.array([[0, 0, 0, 0]]), np.random.randn(10, 4), 1 / 20, 10 / 20),
        (
            np.array([[0, 0]]),
            np.array([[1, 3], [2, 0], [3, 7]]),
            taper_function,
            3 * taper_function(np.array([[1, 3], [2, 0], [3, 7]])),
        ),
        (
            2 * np.pi * np.array([[1 / 2, 1 / 2, 1 / 2], [1, 1, 1]]),
            np.array([[1, 1, 1], [3, 3, 3], [7, 7, 7]]),
            1 / 10,
            np.array([-3 / 10, 3 / 10]),
        ),
    ],
)
def test_tapered_DFT(k, points, taper, expected):
    tapered_DFT = spe.tapered_DFT(k, points, taper)
    np.testing.assert_almost_equal(tapered_DFT, expected)


@pytest.mark.parametrize(
    "k, points, window_volume, intensity ,expected",
    [
        (np.full((1, 2), 5), np.zeros((6, 2)), 3, 2, 6),
        (np.full((1, 8), 2), np.zeros((6, 8)), 2, 3, 6),
        (
            np.full((1, 2), 2 * np.pi),
            np.ones((6, 2)),
            6,
            1,
            6,
        ),
        (np.ones((1, 5)), np.zeros((8, 5)), 4, 2, 8),
        (np.ones((1, 1)), np.zeros((12, 1)), 12, 1, 12),
    ],
)
def test_scattering_intensity(k, points, window_volume, intensity, expected):
    computed = spe.scattering_intensity(k, points, window_volume, intensity)
    np.testing.assert_almost_equal(computed, expected)
