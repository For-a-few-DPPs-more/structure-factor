import numba_scipy
import numba_scipy.special
import numpy as np
import scipy.special as sc
from numba import njit
from scipy.spatial.distance import pdist
from structure_factor.spatial_windows import BallWindow

from structure_factor.spatial_windows import UnitBallWindow

#! doc done (Diala)
def allowed_k_norm(d, r, n):
    r"""Allowed wavenumbers of Bartlett's isotropic estimator, for a d-dimensional point process observed in a ball window of radius r.

    .. warning::

            This method is only available when the space dimension d is an even number.

    Args:
        d (int): Dimension of the underlying space.
        r (float): Radius of the observation window.
        n (int): Number of required allowed wavenumbers.

    Returns:
        np.array: ``n`` allowed wavenumbers.

    Example:
            .. literalinclude:: code/isotropic_estimator/allowed_k_norm.py
                :language: python

            .. testoutput::

                [0.1915853  0.35077933 0.50867341 0.6661846  0.8235315  0.98079293]


    .. proof:definition::

        The allowed wavenumbers of a realization from a point process :math:`\mathcal{X}` of :math:`\mathbb{R}^d` observed in a ball window :math:`W=B(\mathbf{0}, R)` are the :math:`k` s.t.,

        .. math::
            k \in \left\{ \frac{x}{R} \in \mathbb{R} \text{ s.t. }  J_{d/2}(x)=0 \right \}

    .. seealso::

        :py:func:`~structure_factor.isotropic_estimator.bartlett_estimator`.

    """
    if np.floor(d / 2) != d / 2:
        raise ValueError(
            "Allowed wavenumber could be used only when the dimension of the space `d` is an even number (i.e., d/2 is an integer)."
        )
    return sc.jn_zeros(d / 2, n) / r


#! care about case k=0


def bartlett_estimator(point_pattern, k_norm=None, n_allowed_k_norm=60):
    r"""Compute Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` of the point process (isotropic) encapsulated in  ``point_pattern``.

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of a point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

        k_norm (np.ndarray, optional): n rows of wavenumbers where the estimator is to be evaluated. If ``k_norm=None`` (recommended), the observation window is a BallWindow, and the space's dimension is an even number, the estimator will be evaluated on the corresponding set of allowed wavenumbers; In this case, the parameters ``n_allowed_k_norm`` allows to specify the number of allowed wavenumbers. Defaults to None.

        n_allowed_k_norm (int, optional): Number of allowed wavenumbers to be used. Option available when ``k_norm=None``. See :py:func:`~structure_factor.isotropic_estimator.allowed_k_norm`. Defaults to 60.

    Returns:
        tuple(numpy.ndarray, numpy.ndarray):
                - k_norm: Wavenumber(s) on which Bartlett's isotropic estimator has been evaluated.
                - estimation: Evaluation(s) of Bartlett's isotropic estimator at ``k``.

    Example:
            .. literalinclude:: code/isotropic_estimator/bartlett_estimator.py
                :language: python
                :emphasize-lines: 12-15

    .. proof:definition::

            Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` of a realization :math:`\{\mathbf{x}_i\}_{i=1}^N` of a point process :math:`\mathcal{X}` observed in a ball window :math:`W=B(\mathbf{0}, R)` is defined by,

            .. math::
                \widehat{S}_{\mathrm{BI}}(k) =1 + \frac{ (2\pi)^{d/2} }{\rho \mathcal{L}^d(W) \omega_{d-1}} \sum_{ \substack{j, q =1 \\ j\neq q } }^{N }
                 \frac{1}{(k \|\mathbf{x}_j - \mathbf{x}_q\|_2)^{d/2 - 1}}
                J_{d/2 - 1}(k \|\mathbf{x}_j - \mathbf{x}_q\|_2).

            For more details, we refer to :cite:`DGRR:22`, (Section 3.2).

    .. seealso::
            :py:class:`~structure_factor.spatial_windows.BallWindow`,
            :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :py:meth:`~structure_factor.structure_factor.StructureFactor.bartlett_isotropic_estimator`, :py:func:`~structure_factor.isotropic_estimator.allowed_k_norm`.
    """
    window = point_pattern.window
    d = window.dimension
    unit_ball = UnitBallWindow(np.zeros(d))

    X = np.atleast_2d(point_pattern.points)
    norm_xi_xj = pdist(X, metric="euclidean")

    # allowed wavenumbers
    if k_norm is None:
        if np.floor(d / 2) != d / 2:
            raise ValueError(
                "Allowed wavenumber could be used only when the dimension of the space `d` is even (i.e., d/2 is an integer). Hint: use the argument `k_norm` to specify the wavenumbers."
            )
        if not isinstance(window, BallWindow):
            raise TypeError(
                "The window must be an instance of BallWindow. Hint: use PointPattern.restrict_to_window."
            )
        k_norm = allowed_k_norm(d=d, r=window.radius, n=n_allowed_k_norm)

    estimation = np.zeros_like(k_norm)
    order = float(d / 2 - 1)
    isotropic_estimator_njit(estimation, k_norm, norm_xi_xj, order)

    surface, volume = unit_ball.surface, window.volume
    rho = point_pattern.intensity
    estimation *= (2 * np.pi) ** (d / 2) / (surface * volume * rho)
    estimation += 1

    return k_norm, estimation


@njit("double(double, double)")
def bessel1_njit(order, x):
    if order == 0.0:
        return sc.j0(x)
    if order == 1.0:
        return sc.j1(x)
    return sc.jv(order, x)


@njit("void(double[:], double[:], double[:], double)")
def isotropic_estimator_njit(result, k_norm, norm_xi_xj, order):
    for k in range(len(k_norm)):
        sum_k = 0.0
        for ij in range(len(norm_xi_xj)):
            k_xi_xj = k_norm[k] * norm_xi_xj[ij]
            tmp = bessel1_njit(order, k_xi_xj)
            if order > 0:
                k_xi_xj = k_xi_xj ** order
                tmp /= k_xi_xj
            sum_k += tmp
        result[k] = 2 * sum_k
