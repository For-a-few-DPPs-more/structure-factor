import numpy as np
import pytest

from hypton.spatial_windows import BallWindow, BoxWindow, UnitBoxWindow
from hypton.utils import get_random_number_generator

##### BallWindow


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


@pytest.mark.parametrize(
    "dimension, seed",
    (
        [2, None],
        [3, None],
        [4, None],
        [10, None],
        [100, None],
    ),
)
def test_center_belongs_to_unit_ball(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    ball = BallWindow(center)
    assert center in ball


@pytest.mark.parametrize("nb_points", [1, 100])
def test_random_points_fall_inside_ball(nb_points):
    center = np.array([0.4, 4, 40])
    radius = 10
    ball = BallWindow(center, radius)
    random_points = ball.rand(nb_points)
    indicator_vector = ball.indicator_function(random_points)
    assert np.all(indicator_vector)


# BoxWindow


@pytest.mark.parametrize(
    "widths, seed",
    (
        [[2], None],
        [[1, 1, np.pi], None],
        [[0.1, 1, 10], None],
    ),
)
def test_volume_box(widths, seed):
    rng = get_random_number_generator(seed)
    a = rng.normal(size=len(widths))
    b = a + np.array(widths)
    box = BoxWindow(np.column_stack((a, b)))
    expected = np.prod(widths)
    np.testing.assert_almost_equal(box.volume, expected)


@pytest.mark.parametrize(
    "dimension, seed",
    (
        [1, None],
        [2, None],
        [3, None],
        [4, None],
        [10, None],
    ),
)
def test_volume_unit_box_is_one(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    box = UnitBoxWindow(dimension, center)
    np.testing.assert_almost_equal(box.volume, 1.0)


@pytest.mark.parametrize(
    "point, expected",
    [
        (np.array([0, 0]), True),
        (np.array([-1, 5]), True),
        (np.array([5, 6]), False),
    ],
)
def test_box_2d_contains_point(point, expected):
    bounds = np.array([[-5, 5], [-5, 5]])
    box = BoxWindow(bounds)
    assert (point in box) == expected


@pytest.mark.parametrize("nb_points", [1, 100])
def test_random_points_fall_inside_box(nb_points):
    bounds = np.array([[-5, 5], [-5, 5]])
    box = BoxWindow(bounds)
    random_points = box.rand(nb_points)
    indicator_vector = box.indicator_function(random_points)
    assert np.all(indicator_vector)
