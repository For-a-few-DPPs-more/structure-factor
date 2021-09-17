import numpy as np
import pytest
from hypton.spatial_windows import BallWindow, BoxWindow


def volume_3d_ball_window(r):
    return 4 / 3 * np.pi * r ** 3


@pytest.mark.parametrize(
    "r, expected",
    [(i, volume_3d_ball_window(i)) for i in [1, 2, 5, 10]],
)
def test_window_volume_ball_3d(r, expected):
    center = np.array([0, 0, 0])
    window = BallWindow(center, r)
    np.testing.assert_almost_equal(window.volume, expected)


@pytest.mark.parametrize(
    "r, point, expected",
    [
        (1, np.array([1, 2, 3]), True),
        (2, np.array([1, 0, 3]), True),
        (1, np.array([5, 8, 6]), False),
    ],
)
def test_indicator_function_ball_3d(r, point, expected):
    center = np.array([1, 2, 3])
    window = BallWindow(center, r)
    assert window.indicator_function(point) == expected


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), True),
        (np.array([-1, 5]), True),
        (np.array([5, 6]), False),
    ],
)
def test_indicator_function_box_2d(point, expected):
    bounds = np.array([[-5, -5], [5, 5]])
    window = BoxWindow(bounds)
    assert window.indicator_function(point) == expected
