from inspect import Parameter
import numpy as np
import pytest

from structure_factor.spatial_windows import (
    BallWindow,
    BoxWindow,
    UnitBallWindow,
    UnitBoxWindow,
    check_cubic_window,
    check_centered_window,
    subwindow_parameter_max,
)
from structure_factor.utils import get_random_number_generator

##### BallWindow

VOLUME_UNIT_BALL = {
    1: 2,
    2: np.pi,
    3: 4 * np.pi / 3,
    4: np.pi ** 2 / 2,
    5: 8 * np.pi ** 2 / 15,
    6: np.pi ** 3 / 6,
}


@pytest.mark.parametrize(
    "dimension, seed",
    (
        (1, None),
        (2, None),
        (3, None),
        (4, None),
        (5, None),
        (6, None),
    ),
)
def test_volume_unit_ball(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    ball = UnitBallWindow(center)
    np.testing.assert_almost_equal(ball.volume, VOLUME_UNIT_BALL[dimension])


@pytest.mark.parametrize(
    "dimension, radius, seed",
    (
        (1, 10, None),
        (2, 10, None),
        (3, 10, None),
        (4, 10, None),
        (5, 10, None),
        (6, 10, None),
    ),
)
def test_volume_ball(dimension, radius, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    ball = BallWindow(center, radius)
    expected = VOLUME_UNIT_BALL[dimension] * radius ** dimension
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
def test_center_belongs_to_ball(dimension, seed):
    rng = get_random_number_generator(seed)
    center = rng.normal(size=dimension)
    ball = BallWindow(center)
    assert center in ball


@pytest.mark.parametrize(
    "point",
    [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
    ],
)
def test_unit_2d_ball_contains_boundary_points(point):
    d = 2
    center = np.zeros(d)
    ball = UnitBallWindow(center)
    assert point in ball


@pytest.mark.parametrize(
    "center, nb_points, seed",
    (
        [[0.4], 1, None],
        [[0.4, 4], 1, None],
        [[0.4, 4, 40], 1, None],
        [[0.4], 100, None],
        [[0.4, 4], 100, None],
        [[0.4, 4, 40], 100, None],
    ),
)
def test_random_points_fall_inside_ball(center, nb_points, seed):
    rng = get_random_number_generator(seed)
    radius = 10
    ball = BallWindow(center, radius)
    random_points = ball.rand(nb_points, seed=rng)
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
    box = UnitBoxWindow(center)
    np.testing.assert_almost_equal(box.volume, 1.0)


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0, 0], True),
        ([-5, -5], True),
        ([-5, 5], True),
        ([5, -5], True),
        ([5, 5], True),
        ([-1, 5], True),
        ([5, 6], False),
    ],
)
def test_box_2d_contains_point(point, expected):
    bounds = [[-5, 5], [-5, 5]]
    box = BoxWindow(bounds)
    assert (point in box) == expected


@pytest.mark.parametrize(
    "bounds, nb_points, seed",
    (
        [[[-5, 5]], 1, None],
        [[[-5, 5], [-5, 5]], 1, None],
        [[[-5, 5], [-5, 5], [-5, 5]], 1, None],
        [[[-5, 5]], 100, None],
        [[[-5, 5], [-5, 5]], 100, None],
        [[[-5, 5], [-5, 5], [-5, 5]], 100, None],
    ),
)
def test_random_points_fall_inside_box(bounds, nb_points, seed):
    rng = get_random_number_generator(seed)
    box = BoxWindow(bounds)
    random_points = box.rand(nb_points, seed=rng)
    indicator_vector = box.indicator_function(random_points)
    assert np.all(indicator_vector)


@pytest.mark.parametrize(
    "bounds",
    (
        [[-5, 5], [0, 10], [-2, 6]],
        [[-2, 2], [-1, 8]],
    ),
)
def test_check_cubic_window_raises_error_if_BoxWindow_not_cubic(bounds):
    box = BoxWindow(bounds)
    with pytest.raises(ValueError):
        check_cubic_window(box)


def test_check_cubic_window_raises_error_if_not_BoxWindow():
    ball = BallWindow(center=[0, 0], radius=1)
    with pytest.raises(TypeError):
        check_cubic_window(ball)


@pytest.mark.parametrize("center, radius", [([0, 0], 5), ([-1, 3], 6)])
def test_convert_to_spatstat_owin_2D_ball_window(center, radius):
    window = BallWindow(center=center, radius=radius)
    bounds = np.add.outer(window.center, [-radius, radius])
    window_r = window.to_spatstat_owin()
    window_r_bounds = np.array([window_r[1], window_r[2]])
    np.testing.assert_array_equal(window_r_bounds, bounds)


@pytest.mark.parametrize("bounds", [([[-5, 5], [-5, 5]]), ([[0, 10], [-2, 1]])])
def test_convert_to_spatstat_owin__2D_box_window(bounds):
    window = BoxWindow(bounds)
    window_r = window.to_spatstat_owin()
    window_r_bounds = np.array([window_r[1], window_r[2]])
    np.testing.assert_array_equal(window_r_bounds, window.bounds)


@pytest.mark.parametrize(
    "window",
    (
        [BoxWindow([[-2, 2]] * 4)],
        [BoxWindow([-2, 2])],
        [BallWindow(center=[0, 0], radius=10)],
        [BallWindow(center=[0], radius=10)],
    ),
)
def test_check_centered_window(window):
    return check_centered_window(window)


@pytest.mark.parametrize(
    "window",
    [
        (BoxWindow([[-1, 2], [-2.5, 3]])),
        (BoxWindow([[-2.5, 3]])),
        (BallWindow(center=[-1, 1], radius=5)),
        (BallWindow(center=[0, 0, 1], radius=4)),
    ],
)
def test_check_centered_window_raises_error(window):
    with pytest.raises(ValueError):
        check_centered_window(window)


@pytest.mark.parametrize(
    "window, type, expected",
    [
        (BoxWindow(bounds=[[-2, 2]] * 4), "BoxWindow", 4),
        (BoxWindow(bounds=[[-6, 6]] * 4), "BallWindow", 6),
        (BallWindow(center=[0, 0], radius=18), "BallWindow", 18),
        (BallWindow(center=[0, 0], radius=18), "BoxWindow", 36 / np.sqrt(2)),
    ],
)
def test_subwindow_parameter_max(window, type, expected):
    result = subwindow_parameter_max(window, type)
    np.testing.assert_equal(result, expected)
