import numpy as np
import pytest

from structure_factor.spatial_windows import BoxWindow
from structure_factor.tapers import BartlettTaper, SineTaper


@pytest.mark.parametrize(
    "window, expected",
    [
        (BoxWindow([[0, 1], [0, 1], [0, 1]]), 1),
        (BoxWindow([[0, 2], [0, 2]]), 1 / 2),
        (BoxWindow([[-1 / 2, 1 / 2], [-1 / 7, 1 / 8]]), 0),
    ],
)
def test_t0(window, expected):
    t0 = BartlettTaper.taper(1, window)
    np.testing.assert_almost_equal(t0, expected)


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
        (np.array([[0, 0]]), BoxWindow([[-1, 1], [-1, 0]]), np.sqrt(2)),
    ],
)
def test_ft_t0(k, window, expected):
    ft_t0 = BartlettTaper.ft_taper(k, window)
    np.testing.assert_almost_equal(ft_t0, expected)


@pytest.mark.parametrize(
    "p, x, window, expected",
    [
        (
            np.array([[0, 0, 0]]),
            np.array([[1, 2, 4], [0, -3.5, 5]]),
            BoxWindow([[-1, 2], [2, 4], [4, 7]]),
            np.array([0, 0]),
        ),
        (
            4,
            np.array([[3], [0], [3 / 4]]),
            BoxWindow([-1, 3]),
            np.array([0, 0, 0.5]),
        ),
        (
            np.array([[4, 4]]),
            np.array([[3, 3], [0, 3], [3 / 4, -3 / 4]]),
            BoxWindow([[-1, 3], [-1, 3]]),
            np.array([0, 0, -1 / 4]),
        ),
    ],
)
def test_sin_taper(p, x, window, expected):
    taper = SineTaper(p)
    t_p = taper.taper(x, window)
    np.testing.assert_almost_equal(t_p, expected)


@pytest.mark.parametrize(
    "p, k, window, expected",
    [
        (
            np.array([[1, 2, 1]]),
            np.array(
                [[np.pi / 2, np.pi / 4, np.pi / 2], [np.pi / 2, np.pi / 4, np.pi / 2]]
            ),
            BoxWindow([[-1, 1], [-2, 2], [-1, 1]]),
            np.array(
                [
                    (-1j * 8 * np.sqrt(2)) / (3 * np.pi),
                    (-1j * 8 * np.sqrt(2)) / (3 * np.pi),
                ]
            ),
        ),
        (2, np.pi / 4, BoxWindow([-2, 2]), (-1j * 8 * np.sqrt(2)) / (3 * np.pi)),
    ],
)
def test_ft_sin_taper(p, k, window, expected):
    taper = SineTaper(p)
    ft_t_p = taper.ft_taper(k, window)
    np.testing.assert_almost_equal(ft_t_p, expected)
