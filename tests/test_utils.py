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
    x_k = np.arange(-max_k, max_k + 1, 1)
    x_k = x_k[x_k != 0]
    X, Y = np.meshgrid(x_k, x_k)
    true_allowed_wave_values = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L
    assert np.equal(
        utils.allowed_wave_values(L, max_k, meshgrid_size), true_allowed_wave_values
    ).all()


@pytest.mark.parametrize(
    "k, points, expected",
    [
        (np.array([[5], [5]]).T, np.array([np.zeros((6)), np.zeros((6))]).T, 6),
        (
            np.array([[2 * np.pi], [2 * np.pi]]).T,
            np.array([np.ones((6)), np.ones((6))]).T,
            6,
        ),
    ],
)
def test_compute_scattering_intensity(k, points, expected):
    np.testing.assert_almost_equal(
        utils.compute_scattering_intensity(k, points), expected
    )
