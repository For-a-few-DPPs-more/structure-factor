import numpy as np
import pytest

import structure_factor.utils as utils


# todo: fix successive np.testing.assert_array_equal calls, should be a single assert
@pytest.mark.parametrize(
    "k, x, y, expected_k_sorted, expected_x_sorted, expected_y_sorted",
    [
        (
            np.array([1, -1, 0, 3, 2]),
            np.array([1, 2, 3, 4, 5]),
            np.array([2, 3, 4, 5, 6]),
            np.array([-1, 0, 1, 2, 3]),
            np.array([2, 3, 1, 5, 4]),
            np.array([3, 4, 2, 6, 5]),
        ),
    ],
)
def test_sort_vectors(k, x, y, expected_k_sorted, expected_x_sorted, expected_y_sorted):
    k_sorted, x_sorted, y_sorted = utils._sort_vectors(k, x, y)
    np.testing.assert_array_equal(k_sorted, expected_k_sorted)
    np.testing.assert_array_equal(x_sorted, expected_x_sorted)
    np.testing.assert_array_equal(y_sorted, expected_y_sorted)


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
        (
            2,
            2 * np.pi,
            1,
            (4, 4),
            np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        ),
    ],
)
def test_allowed_wave_vectors(d, L, max_k, meshgrid_size, result):
    # ? seems like a copy paste from original code
    computed, _ = utils.allowed_wave_vectors(d, L, max_k, meshgrid_size)
    np.testing.assert_array_equal(computed, result)


@pytest.mark.parametrize(
    "k, points, expected",
    [
        (np.full((1, 2), 5), np.zeros((6, 2)), 6),
        (np.full((1, 8), 2), np.zeros((6, 8)), 6),
        (
            np.full((1, 2), 2 * np.pi),
            np.ones((6, 2)),
            6,
        ),
        (np.ones((1, 5)), np.zeros((8, 5)), 8),
        (np.ones((1, 1)), np.zeros((12, 1)), 12),
    ],
)
def test_compute_scattering_intensity(k, points, expected):
    computed = utils.compute_scattering_intensity(k, points)
    np.testing.assert_almost_equal(computed, expected)


@pytest.mark.parametrize(
    "d, k, L, expected",
    [
        (
            3,
            np.array(
                (
                    [2 * np.pi * 3 / 5, 2 * np.pi * 3 / 5, 2 * np.pi * 3 / 5],
                    [2 * np.pi * 2 / 5, 2 * np.pi * 2 / 5, 2 * np.pi * 2 / 5],
                )
            ),
            5,
            np.array([0, 0]),
        ),
        (
            4,
            np.array(
                [
                    [
                        (2 * np.pi) / (2 * 6),
                        (2 * np.pi) / (2 * 6),
                        (2 * np.pi) / (2 * 6),
                        (2 * np.pi) / (2 * 6),
                    ]
                ]
            ),
            6,
            ((2 * 6) / (np.pi * np.sqrt(6))) ** 4,
        ),
    ],
)
def test_H_0(d, k, L, expected):
    H_0 = utils.H_0(d, k, L)
    np.testing.assert_almost_equal(H_0, expected)


@pytest.mark.parametrize(
    "h_0, k, points, expected",
    [
        (0, 2 * np.pi * 7, np.random.randn(20, 1), 0),
        (1 / 20, np.array([[0, 0, 0, 0]]), np.random.randn(10, 4), 10 / 20),
        (
            1 / 10,
            2 * np.pi * np.array([[1 / 2, 1 / 2, 1 / 2], [1, 1, 1]]),
            np.array([[1, 1, 1], [3, 3, 3], [7, 7, 7]]),
            np.array([-3 / 10, 3 / 10]),
        ),
    ],
)
def test_J_0(h_0, k, points, expected):
    J_0 = utils.J_0(h_0, k, points)
    np.testing.assert_almost_equal(J_0, expected)
