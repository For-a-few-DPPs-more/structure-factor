from unittest import result
import numpy as np
import pytest
from structure_factor.data import load_data

from structure_factor.multiscale_estimators import (
    multiscale_estimator_core,
    coupled_sum_estimator,
    _subwindow_param_max,
    subwindows_list,
    _poisson_rv,
    m_threshold,
)
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.point_pattern import PointPattern
from structure_factor.structure_factor import StructureFactor


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


@pytest.mark.parametrize(
    "window, points, k",
    [
        (
            BoxWindow([[-8, 8]] * 3),
            np.array([[np.pi, 1, 3], [2 * np.pi, 4, 2]]),
            np.array([[1, 2, 3]]),
        ),
        (
            BoxWindow([[-10, 8]] * 2),
            np.array([[1, 3], [2 * np.pi, 2]]),
            np.array([[5, 0]]),
        ),
    ],
)
def test_multiscale_estimator_on_one_scale(window, points, k):
    # test with the scattering intensity on one window will give the scattering intensity for one wavevector 3D.

    # PointPattern
    point_pattern = PointPattern(points, window)
    # result
    result = multiscale_estimator_core(
        point_pattern,
        subwindows_list=[window],
        k_list=[k],
        estimator="scattering_intensity",
    )
    # expected
    sf = StructureFactor(point_pattern)
    _, expected = sf.scattering_intensity(k=k)
    expected = [expected] * len(result)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "proba_list, y_list, expected_result",
    [(1, 2, 2), ([1, 2, 3], [2, 4, 1], 2)],
)
def test_coupled_sum_estimator_on_simple_values(proba_list, y_list, expected_result):
    result = coupled_sum_estimator(y_list, proba_list)
    print(result, expected_result)
    np.testing.assert_almost_equal(result, expected_result)


@pytest.mark.parametrize(
    "window, type, param_0, params, expected",
    [
        (
            BoxWindow([[-3, 4], [-2, 2]]),
            "BallWindow",
            None,
            [1],
            [np.pi],
        ),
        (
            BoxWindow([[-3, 4]]),
            "BallWindow",
            None,
            [2],
            [4],
        ),
        (
            BoxWindow([[-6, 4]]),
            "BoxWindow",
            None,
            [2],
            [2],
        ),
        (
            BallWindow(center=[0, 0], radius=6),
            "BoxWindow",
            4,
            None,
            [4 ** 2, 5 ** 2, 6 ** 2, 7 ** 2, 8 ** 2],
        ),
    ],
)
def test_subwindows_volume(window, type, param_0, params, expected):
    subwindows, _ = subwindows_list(
        window=window, type=type, param_0=param_0, params=params
    )
    result = [w.volume for w in subwindows]
    np.testing.assert_equal(result, expected)


# def test_k_list_with_scattering_intensity():
#     estimator = "scattering_intensity"
#     d = 3
#     subwindows_params = [4]
#     expected = [np.full((1, 3), fill_value=2 * np.pi / 4)]
#     result = k_list(estimator, d, subwindows_params)
#     np.testing.assert_equal(result, expected)


def test_m_under_threshold():
    mean_poisson, threshold = 30, 32
    m_list = [_poisson_rv(mean_poisson, threshold) for _ in range(20)]
    result = sum(m > threshold for m in m_list)
    expected = 0
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "window, type, expected",
    [
        (BoxWindow(bounds=[[-2, 6.2]] * 4), "BoxWindow", 8.2),
        (BoxWindow(bounds=[[-2, 6]] * 4), "BallWindow", 4),
        (BallWindow(center=[0, 0], radius=18), "BallWindow", 18),
        (BallWindow(center=[0, 0], radius=18), "BoxWindow", 36 / np.sqrt(2)),
    ],
)
def test_subwindow_param_max(window, type, expected):
    result = _subwindow_param_max(window, type)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "window_min, window_max, expected",
    [
        (
            BallWindow(center=[0, 0, 0], radius=10),
            BallWindow(center=[0, 0, 0], radius=10),
            0,
        ),
        (
            BallWindow(center=[0, 0], radius=10.234),
            BallWindow(center=[0, 0], radius=20.5),
            10,
        ),
        (
            BallWindow(center=[0, 0, 0], radius=4.234),
            BoxWindow(bounds=[[-10, 10]] * 3),
            5,
        ),
        (
            BoxWindow(bounds=[[-10, 10]] * 3),
            BoxWindow(bounds=[[-20, 20]] * 3),
            20,
        ),
        (
            BoxWindow(bounds=[[-2, 4], [-3, 2]]),
            BallWindow(center=[0, 0], radius=8.34),
            6,
        ),
    ],
)
def test_m_threshold(window_min, window_max, expected):
    result = m_threshold(window_min, window_max)
    np.testing.assert_equal(result, expected)
