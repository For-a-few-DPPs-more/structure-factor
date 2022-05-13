"""Collection of point processes at related properties, e.g., intensity, pair correlation function, structure factor.

- :py:class:`~structure_factor.point_processes.HomogeneousPoissonPointProcess`: The homogeneous Poisson point process.

- :py:class:`~structure_factor.point_processes.ThomasPointProcess`: The Thomas point process.

- :py:class:`~structure_factor.point_processes.GinibrePointProcess`: The Ginibre point process.

- :py:func:`~structure_factor.point_processes.mutual_nearest_neighbor_matching`: The matching process of :cite:`KlaLasYog20`.

"""
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree

from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import (
    AbstractSpatialWindow,
    BallWindow,
    BoxWindow,
)
from structure_factor.utils import get_random_number_generator


class HomogeneousPoissonPointProcess(object):
    """`Homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_.

    .. todo::

        list attributes
    """

    def __init__(self, intensity=1.0):
        """Create a `homogeneous Poisson point process <https://en.wikipedia.org/wiki/Poisson_point_process#Spatial_Poisson_point_process>`_ with prescribed (positive) ``intensity`` parameter.

        Args:
            intensity (float, optional): intensity of the homogeneous Poisson point process. Defaults to 1.0.
        """
        if not intensity > 0:
            raise TypeError("intensity argument must be 2positive")
        self._intensity = float(intensity)

    @property
    def intensity(self):
        r"""Return the intensity :math:`\rho_1(r) = \rho` of the homogeneous Poisson point process.

        Returns:
            float: Constant intensity.
        """
        return self._intensity

    @staticmethod
    def pair_correlation_function(r=None):
        r"""Evaluate the pair correlation function :math:`g(r)=1` of the homogeneous Poisson process.

        Args:
            r (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the pair correlation function is evaluated. Since the homogeneous Poisson process is isotropic, a vector of size :math:`n` corresponding to the norm of the :math:`n` points can also be passed. Defaults to None.

        Returns:
            float or numpy.ndarray: ``1.0`` if ``r=None``, otherwise a vector of size :math:`n` with entries equal to ``1.0``.
        """
        val = 1.0
        if r is None:
            return val

        assert r.ndim <= 2
        return np.full((r.shape[0], 1), val)

    @staticmethod
    def structure_factor(k=None):
        r"""Evaluate the structure factor :math:`S(k)=1` of the homogeneous Poisson process.

        Args:
            k (numpy.ndarray): Array of size :math:`n \times d`, where :math:`d` is the ambient dimension and :math:`n` the number of points where the structure factor is evaluated. Since the homogeneous Poisson process is isotropic, a vector of size :math:`n` corresponding to the norm of the :math:`n` points can also be passed. Defaults to None.

        Returns:
            float or numpy.ndarray: ``1.0`` if ``k=None``, otherwise a vector of size :math:`n` with entries equal to ``1.0``.
        """
        val = 1.0
        if k is None:
            return val

        assert k.ndim <= 2
        return np.full(k.shape[0], val)

    def generate_sample(self, window, seed=None):
        r"""Generate an exact sample of the point process restricted to the :math:`d` dimensional `window`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): :math:`d`-dimensional observation window where the points will be generated.

        Returns:
            numpy.ndarray: Array of size :math:`n \times d`, where :math:`n` is the number of points forming the sample.

        Example:
            .. plot:: code/point_processes/poisson_sample.py
                :include-source: True
                :caption:
                :alt: code/point_processes/poisson_sample.py
                :align: center

        .. seealso::

            - :py:mod:`~structure_factor.spatial_windows`
        """
        if not isinstance(window, AbstractSpatialWindow):
            raise TypeError("window argument must be an AbstractSpatialWindow")

        rng = get_random_number_generator(seed)
        rho = self.intensity
        nb_points = rng.poisson(rho * window.volume)
        return window.rand(nb_points, seed=rng)

    def generate_point_pattern(self, window, seed=None):
        """Generate a :py:class:`~structure_factor.point_pattern.PointPattern` of the point process.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Observation window.
            seed (int, optional): Seed to initialize the points generator. Defaults to None.

        Returns:
            :py:class:`~structure_factor.point_pattern.PointPattern`:Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of the point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

        Example:
            .. plot:: code/point_processes/poisson_pp.py
                :include-source: True
                :caption:
                :alt: code/point_processes/poisson_pp.py
                :align: center

        .. seealso::

            - :py:mod:`~structure_factor.spatial_windows`
            - :py:class:`~structure_factor.point_pattern.PointPattern`
        """
        points = self.generate_sample(window=window, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern


class ThomasPointProcess:
    """Homogeneous Thomas point process with Gaussian clusters.

    .. todo::

        list attributes
    """

    def __init__(self, kappa, mu, sigma):
        """Create a homogeneous Thomas point process.

        Args:
            kappa (float): Mean number of clusters per unit volume. Intensity of the parent Poisson point process.

            mu (float): Mean number of points per clusters. Mean of the child Gaussian distribution.

            sigma (float): Standard deviation of the child Gaussian distribution.
        """
        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

        self._intensity = float(kappa * mu)

    @property
    def intensity(self):
        r"""Return the intensity :math:`\rho_1(r) = \kappa \mu` of the Thomas point process.

        Returns:
            float: Constant intensity.
        """
        return self._intensity

    def pair_correlation_function(self, r_norm, d=2):
        r"""Evaluate the pair correlation function of the Thomas point process.

        .. math::

            g(r)
            = 1
            + \kappa (4 \pi \sigma^2)^{d / 2} \exp(-\frac{1}{4\sigma^2} \|r\|^2)

        Args:
            r_norm (numpy.ndarray): Vector of size :math:`n` corresponding to the norm of the :math:`n` points where the pair correlation function is evaluated.

            d (int, optional): Ambient dimension. Defaults to 2.

        Returns:
            numpy.ndarray: Vector of size :math:`n` containing the evaluation of the pair correlation function.
        """
        k = self.kappa
        s2 = self.sigma ** 2

        pcf = np.exp(r_norm ** 2 / (-4 * s2))
        pcf /= k * (4 * np.pi * s2) ** (d / 2)
        pcf += 1.0
        return pcf

    def structure_factor(self, k_norm):
        r"""Evaluate the structure factor of the Thomas point process.

        .. math::

            S(k) = 1 + \mu \exp(- \sigma^2 * \|k\|^2).

        Args:
            k_norm (numpy.ndarray): Vector of size :math:`n` corresponding to the norm of the :math:`n` points where the structure factor is evaluated.

        Returns:
            numpy.ndarray: Vector of size :math:`n` containing the evaluation of the structure factor.
        """
        mu = self.mu
        s2 = self.sigma ** 2
        return 1.0 + mu * np.exp(-s2 * k_norm ** 2)

    def generate_sample(self, window, seed=None):
        r"""Generate an exact sample from the corresponding :py:class:`~structure_factor.thomas_process.ThomasPointProcess` restricted to the :math:`d` dimensional `window`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): :math:`d`-dimensional observation window where the points will be generated.

        Returns:
            numpy.ndarray: Array of size :math:`n \times d`, where :math:`n` is the number of points forming the sample.

        Example:
            .. plot:: code/point_processes/thomas_sample.py
                :include-source: True
                :caption:
                :alt: code/point_processes/thomas_sample.py
                :align: center

        .. seealso::

            - :py:mod:`~structure_factor.spatial_windows`
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

        pp = HomogeneousPoissonPointProcess(self.kappa)
        centers = pp.generate_sample(extended_window, seed=rng)
        n_per_cluster = np.random.poisson(self.mu, size=len(centers))

        d = window.dimension
        s = self.sigma
        points = np.vstack(
            [rng.normal(c, s, (n, d)) for (c, n) in zip(centers, n_per_cluster)]
        )
        mask = window.indicator_function(points)
        return points[mask]

    def generate_point_pattern(self, window, seed=None):
        """Generate a :py:class:`~structure_factor.point_pattern.PointPattern` of the point process.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): Observation window.

            seed (int, optional): Seed to initialize the points generator. Defaults to None.

        Returns:
            :py:class:`~structure_factor.point_pattern.PointPattern`:Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of the point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

        Example:
            .. plot:: code/point_processes/thomas_pp.py
                :include-source: True
                :caption:
                :alt: code/point_processes/thomas_pp.py
                :align: center

        .. seealso::

            - :py:mod:`~structure_factor.spatial_windows`
            - :py:class:`~structure_factor.point_pattern.PointPattern`
        """
        points = self.generate_sample(window=window, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern


class GinibrePointProcess(object):
    """Ginibre point process corresponds to the complex eigenvalues of a standard complex Gaussian matrix.

    .. todo::

        list attributes
    """

    def __init__(self):
        self._intensity = 1.0 / np.pi

    @property
    def intensity(self):
        r"""Return the intensity :math:`\rho_1(r) = \frac{1}{\pi}` of the Ginibre point process.

        Returns:
            float: Constant intensity.
        """
        return self._intensity

    @staticmethod
    def pair_correlation_function(r_norm):
        r"""Evaluate the pair correlation function of the Ginibre point process.

        .. math::

            g(r) = 1 - \exp(- \|r\|^2)

        Args:
            r_norm (numpy.ndarray): Vector of size :math:`n` corresponding to the norm of the :math:`n` points where the pair correlation function is evaluated.

        Returns:
            numpy.ndarray: Vector of size :math:`n` containing the evaluation of the pair correlation function.
        """
        return 1.0 - np.exp(-(r_norm ** 2))

    @staticmethod
    def structure_factor(k_norm):
        r"""Evaluate the structure factor of the Ginibre point process.

        .. math::

            S(k) = 1 - \exp(- \frac{1}{4} \|k\|^2).

        Args:
            k_norm (numpy.ndarray): Vector of size :math:`n` corresponding to the norm of the :math:`n` points where the structure factor is evaluated.

        Returns:
            numpy.ndarray: Vector of size :math:`n` containing the evaluation of the structure factor.
        """
        return 1.0 - np.exp(-0.25 * (k_norm ** 2))

    def generate_sample(self, window, n=None, seed=None):
        r"""Generate a sample of the Ginibre point process made of ``n`` points, inside the observation ``window``.

        This is done by computing the eigenvalues of a random matrix :math:`G` of size :math:`n \times n`, filled with i.i.d. standard complex Gaussian variables, i.e.,

        .. math::

            G_{ij} = \frac{1}{\sqrt{2}} (X_{ij} + \mathbf{i} Y_{ij}).

        Args:
            window (:py:class:`~structure_factor.spatial_windows.BallWindow`): :math:`2`-dimensional centered ball window where the points will be generated.

            n (int, optional): Number of points of the output sample. If ``n`` is None (default), it is set to the integer part of :math:`\rho |W| = \frac{1}{\pi} |W|`. Defaults to None.

        Returns:
            numpy.ndarray: Array of size :math:`n \times 2`, representing the :math:`n` points forming the sample.

        Example:
            .. plot:: code/point_processes/ginibre_sample.py
                :include-source: True
                :caption:
                :alt: code/point_processes/ginibre_sample.py
                :align: center

        .. seealso::

            - :py:class:`~structure_factor.spatial_windows.BallWindow`
        """
        if not isinstance(window, BallWindow):
            raise ValueError("The window should be a 2-d centered BallWindow.")
        if window.dimension != 2:
            raise ValueError("The window should be a 2-d window.")
        if not np.all(np.equal(window.center, 0.0)):
            raise ValueError("The window should be a centered window.")

        if n is None:
            n = int(window.volume * self.intensity)
        assert isinstance(n, int)

        rng = get_random_number_generator(seed)

        A = np.zeros((n, n), dtype=complex)
        A.real = rng.standard_normal((n, n))
        A.imag = rng.standard_normal((n, n))
        eigvals = la.eigvals(A) / np.sqrt(2.0)

        points = np.vstack((eigvals.real, eigvals.imag)).T
        mask = window.indicator_function(points)

        return points[mask]

    def generate_point_pattern(self, window, n=None, seed=None):
        r"""Generate a :math:`2`-dimensional :py:class:`~structure_factor.point_pattern.PointPattern` of the point process, with a centered :py:class:`~structure_factor.spatial_windows.BallWindow`.

        Args:
            window (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`): :math:`2`-dimensional observation centered :py:class:`~structure_factor.spatial_windows.BallWindow`.

            n (int, optional): Number of points of the output sample. If ``n`` is None (default), it is set to the integer part of :math:`\rho |W| = \frac{1}{\pi} |W|`. Defaults to None.

            seed (int, optional): Seed to initialize the points generator. Defaults to None.

        Returns:
            :py:class:`~structure_factor.point_pattern.PointPattern`:Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of the point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

        Example:
            .. plot:: code/point_processes/ginibre_pp.py
                :include-source: True
                :caption:
                :alt: code/point_processes/ginibre_pp.py
                :align: center

        .. seealso::

            - :py:class:`~structure_factor.spatial_windows.BallWindow`
            - :py:class:`~structure_factor.point_pattern.PointPattern`
        """
        points = self.generate_sample(window=window, n=n, seed=seed)
        point_pattern = PointPattern(
            points=points, window=window, intensity=self.intensity
        )
        return point_pattern


def mutual_nearest_neighbor_matching(X, Y, **KDTree_params):
    r"""Match the set of points ``X`` with a subset of points from ``Y`` based on mutual nearest neighbor matching :cite:`KlaLasYog20`. It is assumed that :math:`|X| \leq |Y|` and that each point in ``X``, resp. ``Y``, can have only one nearest neighbor in ``Y``, resp. ``X``.

    The matching routine involves successive 1-nearest neighbor sweeps performed by `scipy.spatial.KDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_ with the euclidean distance.

    Args:
        X (numpy.ndarray): Array of size (m, d) collecting points to be matched with a subset of points from ``Y``.
        Y (numpy.ndarray): Array of size (n, d) of points satisfying :math:`m \leq n`.

    Keyword Args:
        see (documentation): keyword arguments of `scipy.spatial.KDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html>`_.

    .. note::

        The ``boxsize`` keyword argument can be used **only** when points belong to a box :math:`\prod_{i=1}^{d} [0, L_i)` (upper boundary excluded). It allows to consider periodic boundaries, i.e., the toroidal distance is used for searching for nearest neighbors.

    Returns:
        numpy.ndarray: vector of indices ``matches`` such that ``X[i]`` is matched to ``Y[matches[i]]``.

    Example:
        .. plot:: code/point_processes/kly_matching.py
            :include-source: True
            :caption: KLY  matching
            :alt: code/point_processes/kly_matching.py
            :align: center
    """
    if not (X.ndim == Y.ndim == 2):
        raise ValueError(
            "X and Y must be 2d numpy arrays with respective size (m, d) and (n, d), where d is the ambient dimension."
        )
    if X.shape[0] > Y.shape[0]:
        raise ValueError(
            "The sets of points represented by X and Y must satisfy |X| <= |Y|."
        )

    m, n = X.shape[0], Y.shape[0]
    idx_X_unmatched = np.arange(m, dtype=int)
    idx_Y_unmatched = np.arange(n, dtype=int)
    matches = np.zeros(m, dtype=int)

    for _ in range(m):  # at most |X| nearest neighbor sweeps are performed

        X_ = X[idx_X_unmatched]
        Y_ = Y[idx_Y_unmatched]

        knn = KDTree(Y_, **KDTree_params)
        X_to_Y = knn.query(X_, k=1, p=2)[1]  # p=2, i.e., euclidean distance

        knn = KDTree(X_, **KDTree_params)
        Y_to_X = knn.query(Y_, k=1, p=2)[1]

        identity = range(len(idx_X_unmatched))
        mask_X = np.equal(Y_to_X[X_to_Y], identity)

        matches[idx_X_unmatched[mask_X]] = idx_Y_unmatched[X_to_Y[mask_X]]

        if np.all(mask_X):  # all points from X got matched
            break

        idx_X_unmatched = idx_X_unmatched[~mask_X]
        mask_Y = np.full(len(idx_Y_unmatched), True, dtype=bool)
        mask_Y[X_to_Y[mask_X]] = False
        idx_Y_unmatched = idx_Y_unmatched[mask_Y]

    return matches
