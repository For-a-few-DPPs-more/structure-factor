import numpy as np
import pytest
from structure_factor.data import load_data

from structure_factor.multiscale_estimators import (
    multiscale_estimator_core,
    coupled_sum_estimator,
    subwindows_list,
    _poisson_rv,
    m_threshold,
    multiscale_estimator,
)
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.point_pattern import PointPattern
from structure_factor.structure_factor import StructureFactor
from structure_factor.tapers import multi_sinetaper_grid


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


@pytest.mark.parametrize(
    "window, type, param_0, params, expected",
    [
        (
            BoxWindow([[-3, 3], [-2, 2]]),
            "BallWindow",
            None,
            [1],
            [np.pi],
        ),
        (
            BoxWindow([[-4, 4]]),
            "BallWindow",
            None,
            [2],
            [4],
        ),
        (
            BoxWindow([[-6, 6]]),
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
        window=window, subwindows_type=type, param_0=param_0, params=params
    )
    result = [w.volume for w in subwindows]
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "estimator, subwindows_type, tapers, expected",
    [
        ("scattering_intensity", "BoxWindow", None, -5.9259),
        (
            "tapered_estimator",
            "BoxWindow",
            multi_sinetaper_grid(d=2, p_component_max=2),
            -6.6391,
        ),
        ("bartlett_isotropic_estimator", "BallWindow", None, 2.1905),
    ],
)
def test_multiscale_estimator_on_real_data(
    ginibre_pp, estimator, subwindows_type, tapers, expected
):
    point_pattern = ginibre_pp
    window = BallWindow(center=[0, 0], radius=20)
    subwindows, k = subwindows_list(window, subwindows_type=subwindows_type, param_0=3)
    result = multiscale_estimator(
        point_pattern,
        estimator=estimator,
        k_list=k,
        subwindows_list=subwindows,
        mean_poisson=None,
        m=7,
        proba_list=[0.11, 0.02, 0.143, 0.4, 0.2, 0.06, 0.07],
        tapers=tapers,
    )
    np.testing.assert_almost_equal(result, expected, decimal=4)


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
def test_multiscale_estimator_core_on_one_scale(window, points, k):
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


def test_m_under_threshold():
    mean_poisson, threshold = 30, 32
    m_list = [_poisson_rv(mean_poisson, threshold) for _ in range(20)]
    result = sum(m > threshold for m in m_list)
    expected = 0
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
            BoxWindow(bounds=[[-2.5, 2.5], [-5, 5]]),
            BallWindow(center=[0, 0], radius=8.34),
            6,
        ),
    ],
)
def test_m_threshold(window_min, window_max, expected):
    result = m_threshold(window_min, window_max)
    np.testing.assert_equal(result, expected)
