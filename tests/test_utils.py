import numpy as np
import pytest

import structure_factor.utils as utils
from structure_factor.spatial_windows import BoxWindow


def test_pair_correlation_function_ginibre():
    r = np.array([[0], [1], [10 ^ 5]])
    pcf = utils.pair_correlation_function_ginibre(r)
    expected = np.array([[0], [1 - 1 / np.exp(1)], [1]])
    np.testing.assert_array_equal(pcf, expected)


def test_structure_factor_ginibre():
    k = np.array([[0], [1], [10 ^ 5]])
    sf = utils.structure_factor_ginibre(k)
    expected = np.array([[0], [1 - 1 / np.exp(1 / 4)], [1]])
    np.testing.assert_array_equal(sf, expected)


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
            np.array([[2 * np.pi], [2 * np.pi], [2 * np.pi]]),
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
            np.array([[2 * np.pi], [2 * np.pi]]),
            1,
            (4, 4),
            np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]]),
        ),
    ],
)
def test_allowed_wave_vectors(d, L, max_k, meshgrid_size, result):
    # ? seems like a copy paste from original code
    computed = utils.allowed_wave_vectors(d, L, max_k, meshgrid_size)
    np.testing.assert_array_equal(computed, result)
