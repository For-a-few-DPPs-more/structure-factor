from structure_factor.spatial_windows import (
    AbstractSpatialWindow,
    BallWindow,
    BoxWindow,
)
from structure_factor.utils import get_random_number_generator
import structure_factor.utils as utils
import numpy as np


class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_."""

    def __init__(self, intensity=1.0):
        """Create a `homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_ with prescribed (positive) ``intensity`` parameter.

        :param intensity: Constant intensity parameter of the homogeneous Poisson point process, defaults to 1.0
        :type intensity: real, optional
        """
        if not intensity > 0:
            raise TypeError("intensity argument must be 2positive")
        self.intensity = intensity

    def structure_factor(self, k):
        """Structure factor of the Poisson point process

        Args:
            k (np.array): Points to evaluate on.

        Returns:
            np.array: Structure factor of Poisson process evaluated on `k`.
        """
        return np.ones(k.shape[0])

    def pair_correlation_function(self, r):
        """Pair correlation function of the Poisson point process

        Args:
            r (np.array): Points to evaluate on.

        Returns:
            np.array: Structure factor of Poisson process evaluated on `k`.
        """
        return np.ones(r.shape[0])

    def generate_sample(self, window, seed=None):
        r"""Generate an exact sample from the corresponding :py:class:`~structure_factor.homogeneous_poisson_process.HomogeneousPoissonPointProcess` restricted to the :math:`d` dimensional `window`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): Observation window where the points will be generated.

        Returns:
            numpy.ndarray: of size :math:`n \times d`, where :math:`n` is the number of points forming the sample.

        .. plot:: code/poisson_sample_example.py
            :include-source: True
            :caption:
            :alt: alternate text
            :align: center
        """
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)
        nb_points = rng.poisson(self.intensity * window.volume)
        return window.rand(nb_points, seed=rng)

    def generate_list_of_samples(self, s, window):
        sample_list = []
        for i in range(0, s):
            sample = self.generate_sample(window)
            sample_list.append(sample)
        return sample_list


class ThomasProintProcess(object):
    """Homogeneous Thomas point process."""

    def __init__(self, intensity_parent, mu, sigma):
        """Create a homogeneous Thomas point process.

        Args:
            intensity_parent (float): Mean number of clusters per unit volume.
            mu (float): Mean number of points per clusters.
            sigma (float): Standard deviation of the gaussian clusters.
        """
        self.intensity_parent = intensity_parent
        self.mu = mu
        self.sigma = sigma

    @property
    def intensity(self):
        return self.intensity_parent * self.mu

    # todo repasse over this fction semms stranger
    # todo remove dim and add d in int?
    def pair_correlation_function(self, r, dim):
        if isinstance(r, np.ndarray) and r.ndim > 1:
            assert r.ndim == 2  # why??
            assert r.shape[1] == dim

        k = self.intensity_parent
        s2 = self.sigma ** 2

        pcf = np.exp(r ** 2 / (-4 * s2))
        pcf /= k * (4 * np.pi * s2) ** (dim / 2)
        pcf += 1.0
        return pcf

    def structure_factor(self, k):
        norm_k = utils.norm_k(k)
        mu = self.mu
        s2 = self.sigma ** 2
        return 1.0 + mu * np.exp(-s2 * norm_k ** 2)

    def generate_sample(self, window, seed=None):
        r"""Generate an exact sample from the corresponding :py:class:`~structure_factor.thomas_process.ThomasPointProcess` restricted to the :math:`d` dimensional `window`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): Observation window where the points will be generated.

        Returns:
            numpy.ndarray: of size :math:`n \times d`, where :math:`n` is the number of points forming the sample.
        """
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)

        tol = 6 * self.sigma
        if isinstance(window, BoxWindow):
            extended_bounds = window.bounds + np.array([-tol, +tol])
            extended_window = BoxWindow(extended_bounds)
        elif isinstance(window, BallWindow):
            exented_radius = window.radius + tol
            extended_window = BallWindow(window.center, exented_radius)

        pp = HomogeneousPoissonPointProcess(self.intensity_parent)
        centers = pp.generate_sample(extended_window, seed=rng)
        n_per_cluster = np.random.poisson(self.mu, size=len(centers))

        d = window.dimension
        s = self.sigma
        points = np.vstack(
            [rng.normal(c, s, (n, d)) for (c, n) in zip(centers, n_per_cluster)]
        )
        return points[window.indicator_function(points)]

    def generate_list_of_samples(self, s, window):
        sample_list = []
        for i in range(0, s):
            sample = self.generate_sample(window)
            sample_list.append(sample)
        return sample_list

    # todo move all structure factor and pair correlation functions here for ginibre add note for the simulation to use DPPY

    # todo add plot function similar to taht of point_pattern
