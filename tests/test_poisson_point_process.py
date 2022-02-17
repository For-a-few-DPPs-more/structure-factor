import numpy as np
import pytest
from scipy.stats import norm

from structure_factor.point_processes import HomogeneousPoissonPointProcess
from structure_factor.spatial_windows import BoxWindow


@pytest.mark.parametrize(
    "nb_samples, rho, W",
    (
        [
            10_000,
            100,
            BoxWindow(np.array([[0, 2], [0, 2]])),
        ],
    ),
)
def test_intensity_estimation_of_poisson_process(nb_samples, rho, W):
    """Estimate the intensity ``rho`` as mean number of points divided by the volume of the observation window ``W`` from ``nb_samples`` and check that the intensity falls inside a confidence interval constructed from the following central limit theorem-like result.

    As W -> R^d, we have

    2 sqrt(vol(W)) (sqrt(rho_hat) − sqrt(rho)) ~ N(0,1)
    """
    hpp = HomogeneousPoissonPointProcess(rho)
    nb_points = [len(hpp.generate_sample(W)) for _ in range(nb_samples)]
    rho_hat = np.mean(nb_points) / W.volume
    # mean_rho_hat = rho
    # var_rho_hat = rho / W.volume
    alpha = 0.05
    center = np.sqrt(rho_hat)
    z_a2 = -norm.ppf(alpha / 2)
    width_2 = z_a2 / (2 * np.sqrt(W.volume))
    assert (center - width_2) <= np.sqrt(rho) <= (center + width_2)
