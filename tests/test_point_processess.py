import numpy as np
import pytest
from scipy.stats import norm

from structure_factor.point_processes import (
    GinibrePointProcess,
    HomogeneousPoissonPointProcess,
    ThomasPointProcess,
)
from structure_factor.spatial_windows import BallWindow, BoxWindow


@pytest.mark.parametrize(
    "point_process, nb_samples, W",
    (
        [
            HomogeneousPoissonPointProcess(100),
            10_000,
            BoxWindow(np.array([[0, 2], [0, 2]])),
        ],
        [
            ThomasPointProcess(kappa=1 / (2 * np.pi), mu=2 * np.pi, sigma=2),
            10_000,
            BallWindow(radius=3, center=[1, 0]),
        ],
    ),
)
def test_intensity_estimation_of_poisson_process(point_process, nb_samples, W):
    """Estimate the intensity ``rho`` as mean number of points divided by the volume of the observation window ``W`` from ``nb_samples`` and check that the intensity falls inside a confidence interval constructed from the following central limit theorem-like result.

    As W -> R^d, we have

    2 sqrt(vol(W)) (sqrt(rho_hat) − sqrt(rho)) ~ N(0,1)
    """

    rho = point_process.intensity
    nb_points = [len(point_process.generate_sample(W)) for _ in range(nb_samples)]
    rho_hat = np.mean(nb_points) / W.volume
    # mean_rho_hat = rho
    # var_rho_hat = rho / W.volume
    alpha = 0.05
    center = np.sqrt(rho_hat)
    z_a2 = -norm.ppf(alpha / 2)
    width_2 = z_a2 / (2 * np.sqrt(W.volume))
    assert (center - width_2) <= np.sqrt(rho) <= (center + width_2)


def test_pair_correlation_function_ginibre():
    r = np.array([[0], [1], [10 ^ 5]])
    pcf = GinibrePointProcess.pair_correlation_function(r)
    expected = np.array([[0], [1 - 1 / np.exp(1)], [1]])
    np.testing.assert_array_equal(pcf, expected)


def test_structure_factor_ginibre():
    k = np.array([[0], [1], [10 ^ 5]])
    sf = GinibrePointProcess.structure_factor(k)
    expected = np.array([[0], [1 - 1 / np.exp(1 / 4)], [1]])
    np.testing.assert_array_equal(sf, expected)
