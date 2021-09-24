import numpy as np
import pytest

from hypton.spatial_windows import BallWindow, BoxWindow

##### BallWindow


@pytest.mark.parametrize(
    "center",
    [
        np.random.randn(2),
        np.random.randn(3),
        np.random.randn(4),
        np.random.randn(10),
        np.random.randn(100),
    ],
)
def test_center_belongs_to_unit_ball(center):
    ball = BallWindow(center)
    assert ball.indicator_function(center)


@pytest.mark.parametrize(
    "dimension, radius, factor",
    (
        (1, 10, 2),
        (2, 10, np.pi),
        (3, 10, 4 * np.pi / 3),
        (4, 10, np.pi ** 2 / 2),
        (5, 10, 8 * np.pi ** 2 / 15),
        (6, 10, np.pi ** 3 / 6),
    ),
)
def test_volume_ball(dimension, radius, factor):
    center = np.zeros(dimension)
    ball = BallWindow(center, radius)
    expected = factor * radius ** dimension
    np.testing.assert_almost_equal(ball.volume, expected)


# BoxWindow


@pytest.fixture
def example_box_window():
    bounds = np.array([[-5, -5], [5, 5]])
    return BoxWindow(bounds)


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), True),
        (np.array([-1, 5]), True),
        (np.array([5, 6]), False),
    ],
)
def test_indicator_function_box_2d(example_box_window, point, expected):
    box = example_box_window
    assert box.indicator_function(point) == expected
