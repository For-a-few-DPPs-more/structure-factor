import numpy as np
import pytest

import hypton.utils as utils


@pytest.mark.parametrize(
    "L, max_k, meshgrid_size",
    [
        (2 * np.pi, 4, 2 * 4 + 1),
        (2 * np.pi, 10, 2 * 10 + 1),
    ],
)
def test_allowed_wave_values(L, max_k, meshgrid_size):
    # ? seems like a copy paste from original code
    x_k = np.arange(-max_k, max_k + 1, 1)
    x_k = x_k[x_k != 0]
    X, Y = np.meshgrid(x_k, x_k)
    expected = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L
    computed = utils.allowed_wave_values(L, max_k, meshgrid_size)
    np.testing.assert_array_equal(computed, expected)


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
