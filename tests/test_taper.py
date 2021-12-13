import numpy as np
import pytest

from structure_factor.spatial_windows import BoxWindow
from structure_factor.tapers import BartlettTaper, sin_taper


@pytest.mark.parametrize(
    "window, expected",
    [(BoxWindow([[0, 1], [0, 1], [0, 1]]), 1), (BoxWindow([[0, 2], [0, 2]]), 1 / 2)],
)
def test_h0(window, expected):
    h0 = BartlettTaper.taper(1, window)
    np.testing.assert_almost_equal(h0, expected)


@pytest.mark.parametrize(
    "k, window, expected",
    [
        (
            np.array(
                (
                    [2 * np.pi * 3 / 5, 2 * np.pi * 3 / 5, 2 * np.pi * 3 / 5],
                    [2 * np.pi * 2 / 5, 2 * np.pi * 2 / 5, 2 * np.pi * 2 / 5],
                )
            ),
            BoxWindow(bounds=[[0, 5], [2, 7], [-1, 4]]),
            np.array([0, 0]),
        ),
        (
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
            BoxWindow(bounds=[[0, 6], [-6, 0], [-1, 5], [3, 9]]),
            ((2 * 6) / (np.pi * np.sqrt(6))) ** 4,
        ),
        (np.array([np.pi, 2.8]), BoxWindow(bounds=[[2, 4], [7, 8]]), 0),
    ],
)
def test_ft_h0(k, window, expected):
    H_0 = BartlettTaper.ft_taper(k, window)
    np.testing.assert_almost_equal(H_0, expected)


@pytest.mark.parametrize(
    "p, x, window, expected",
    [
        (
            np.array([[0, 0, 0]]),
            np.array([[1, 2, 4], [0, -3.5, 5]]),
            BoxWindow([[-1, 2], [2, 4], [4, 7]]),
            np.array([0, 0]),
        ),
        (4, np.array([[3], [0], [3 / 4]]), BoxWindow([-1, 5]), np.array([0, 0, 1])),
        (
            np.array([[4, 4]]),
            np.array([[3, 3], [0, 3], [3 / 4, -3 / 4]]),
            BoxWindow([[-1, 5], [-1, 5]]),
            np.array([0, 0, -1]),
        ),
    ],
)
def test_sin_taper(p, x, window, expected):
    h_p = sin_taper(p, x, window)
    np.testing.assert_almost_equal(h_p, expected)
