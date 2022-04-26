from tkinter.tix import Balloon
import numpy as np
import pytest
from structure_factor.data import load_data

from structure_factor.multiscale_estimators import (
    multiscale_estimator,
    coupled_sum_estimator,
    _subwindows,
    _k_list,
    _m_list,
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
    result = multiscale_estimator(
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
    "window, type, param_0, params, to_test, expected",
    [
        (
            BoxWindow([[-3, 4], [-2, 2]]),
            "Ball",
            None,
            [1],
            "subwindows",
            [np.pi],
        ),
        (
            BoxWindow([[-3, 4]]),
            "Ball",
            None,
            [2],
            "subwindows",
            [4],
        ),
        (
            BoxWindow([[-6, 4]]),
            "Box",
            None,
            [2],
            "subwindows",
            [2],
        ),
        (
            BallWindow(center=[0, 0], radius=6),
            "Box",
            4,
            None,
            "subwindows",
            [4 ** 2, 5 ** 2, 6 ** 2, 7 ** 2],
        ),
        (
            BallWindow(center=[0, 0], radius=6),
            "Ball",
            4,
            None,
            "params",
            [4, 5],
        ),
    ],
)
def test_subwindows(window, type, param_0, params, to_test, expected):
    subwindows, params = _subwindows(window, type, param_0, params)
    if to_test == "params":
        result = params
    else:
        result = [w.volume for w in subwindows]
    np.testing.assert_equal(result, expected)


def test_k_list_with_scattering_intensity():
    estimator = "scattering_intensity"
    d = 3
    subwindows_params = [4]
    expected = [np.full((1, 3), fill_value=2 * np.pi / 4)]
    result = _k_list(estimator, d, subwindows_params)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "to_test",
    [("length"), ("threshold")],
)
def test_m_list_result(to_test):
    mean_poisson, nb_m, threshold = 30, 20, 32
    m_list = _m_list(mean_poisson, nb_m, threshold)
    if to_test == "length":
        result = len(m_list)
        expected = nb_m
    elif to_test == "threshold":
        result = sum(m > threshold for m in m_list)
        expected = 0
    np.testing.assert_equal(result, expected)
