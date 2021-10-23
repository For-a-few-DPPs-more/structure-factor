import numpy as np
import pytest

import structure_factor.utils as utils


@pytest.mark.parametrize(
    "d, L, max_k, meshgrid_size, result",
    [
        (
            1,
            2 * np.pi,
            4,
            None,
            np.array([[-4], [-3], [-2], [-1], [1], [2], [3], [4]]),
        ),
        (
            3,
            2 * np.pi,
            2,
            (4, 2, 2),
            np.array(
                [
                    [-2.0, -2.0, -2.0],
                    [-2.0, -2.0, 2.0],
                    [-1.0, -2.0, -2.0],
                    [-1.0, -2.0, 2.0],
                    [1.0, -2.0, -2.0],
                    [1.0, -2.0, 2.0],
                    [2.0, -2.0, -2.0],
                    [2.0, -2.0, 2.0],
                    [-2.0, 2.0, -2.0],
                    [-2.0, 2.0, 2.0],
                    [-1.0, 2.0, -2.0],
                    [-1.0, 2.0, 2.0],
                    [1.0, 2.0, -2.0],
                    [1.0, 2.0, 2.0],
                    [2.0, 2.0, -2.0],
                    [2.0, 2.0, 2.0],
                ]
            ),
        ),
    ],
)
def test_allowed_wave_values(d, L, max_k, meshgrid_size, result):
    # ? seems like a copy paste from original code
    computed = utils.allowed_wave_values(d, L, max_k, meshgrid_size)
    np.testing.assert_array_equal(computed, result)


@pytest.mark.parametrize(
    "k, points, expected",
    [
        (np.full((1, 2), 5), np.full((6, 2), 0), 6),
        (np.full((1, 8), 2), np.full((6, 8), 0), 6),
        (
            np.full((1, 2), 2 * np.pi),
            np.full((6, 2), 1),
            6,
        ),
    ],
)
def test_compute_scattering_intensity(k, points, expected):
    computed = utils.compute_scattering_intensity(k, points)
    np.testing.assert_almost_equal(computed, expected)
