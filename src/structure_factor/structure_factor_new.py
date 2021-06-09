import numpy as np
import scipy.interpolate as interpolate
import pandas as pd
import rpy2.robjects as robjects

from structure_factor.utils import compute_scattering_intensity
from structure_factor.transforms import RadiallySymmetricFourierTransform
from structure_factor.spatstat_interface import SpatstatInterface


class StructureFactor:
    """Implementation of various estimators of the structure factor of a 2 dimensional point process.
    # todo add link to definition
    """

    def __init__(self, points, intensity):
        r"""
        Args:
            points: :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
            intensity: intensity of the underlying point process.

        # todo treat the case where the intensity is not provided
        # todo consider passing the window where the points were observed to avoid radius in compute_pcf(self, radius...
        # todo est ce qu'on fait une class pointPatern  qui contient le window et les data et on le coupe pour que scattering intensitymarche?
        """
        dimension = points.shape[1]
        assert points.ndim == 2 and dimension == 2
        self.dimension = dimension
        self.points = points
        self.intensity = intensity

    def compute_scattering_intensity(self, L, maximum_wave, meshgrid_size=None):
        # todo replace the link below to the link of our future paper :).
        r"""Compute the ensemble estimator of the scattering intensity described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5).

        .. math::

            SI(k) =
            \left\lvert
                \sum_{x \in \mathcal{X}}
                    \exp(- i \left\langle k, x \right\rangle)
            \right\rvert^2

        Note: This estimation converges to the structure factor in the thermodynamic limits.

        Notes:  The data should be simulated inside a cube. # todo what is data ?
        # todo self.data expliquer qui est data
                The allowed values of wave vectors are the points of the dual of the lattice having fundamental cell the cubic window .
                This is represented inside wave_vectors defined as :
                wave_vectors = 2*pi*k_vector/L where, k_vector is a vector of integer from 1 into maximum_k, and L in the length side of the cubic window.

        Args:
            L (int): length of the square that contains the data.
            maximum_wave (int): maximum of wave vector
            meshgrid_size (int): if the requested evaluation is on a meshgrid,  then meshgrid_size is the number of wave vector in each row of the meshgrid. Defaults to None.

        Returns:
            :math:`\left\lVert k \right\rVert, SI(K)`, the norm of the wave vectors :math:`k` and the estimation of the scattering intensity evaluated at :math:`k`.
        """
        maximum_k = np.floor(maximum_wave * L / (2 * np.pi * np.sqrt(2)))
        if meshgrid_size is None:
            x = np.linspace(1, maximum_k, int(maximum_k))
            k_ = np.column_stack((x, x))
        else:
            x_grid = np.linspace(0, maximum_k, int(meshgrid_size))
            X, Y = np.meshgrid(x_grid, x_grid)
            k_ = np.column_stack((X.ravel(), Y.ravel()))

        k_norm = np.linalg.norm(k_, axis=1)
        si = compute_scattering_intensity(k_, self.points)

        if meshgrid_size is not None:
            k_norm = k_norm.reshape(X.shape)
            si = si.reshape(X.shape)

        return k_norm, si

    def compute_pcf(self, radius, method, install_spatstat=False, **params):
        """Estimate the pair correlation function of ``self.points`` observed in a disk window centered at the origin with radius ``radius`` using spatstat ``spastat.core.pcf_ppp`` or ``spastat.core.pcf_fv`` functions according to ``method`` with the corresponding parameters ``params``.

        # todo consider adding the window where points were observed at __init__ to avoid radius argument.
        # todo expliciter le radius fais quoi

        radius:
        method: "ppp" or "fv" referring to ``spastat.core.pcf.ppp`` or ``spastat.core.pcf.fv`` functions for estimating the pair correlation function.
        install_spatstat: [description], defaults to False
        params:
            - method = 'ppp'
                - dict(``spastat.core.pcf.ppp`` parameters)
            - method = 'fv'
                - dict(
                    Kest=dict(``spastat.core.Kest`` parameters),
                    fv=dict(``spastat.core.pcf.fv`` parameters)
                )

        Return dict representing the DataFrame output of `spastat.core.pcf.ppp`` or ``spastat.core.pcf.fv`` functions.

        .. seealso::

            - `pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_
            - `pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_
        """
        assert method in ("ppp", "fv")

        # core, geom and other subpackages are updated if install_spatstat
        spatstat = SpatstatInterface(update=install_spatstat)
        spatstat.import_package("core", "geom", update=False)

        window = spatstat.geom.disc(radius)

        x = robjects.vectors.FloatVector(self.points[:, 0])
        y = robjects.vectors.FloatVector(self.points[:, 1])
        data = spatstat.geom.ppp(x, y, window=window)

        if method == "ppp":
            pcf = spatstat.core.pcf_ppp(data, **params)

        if method == "fv":
            params_Kest = params.get("Kest", dict())
            k_ripley = spatstat.core.Kest(data, **params_Kest)
            params_fv = params.get("fv", dict())
            pcf = spatstat.core.pcf_fv(k_ripley, **params_fv)

        return pd.DataFrame(np.array(pcf).T, columns=pcf.names)

    # todo faire une méthod pour cleaner les data "import pandas as pd approx_pcf_gin.replace([np.inf, -np.inf], np.nan, inplace=True) cleaned_pd_pcf = pd.DataFrame.from_records(approx_pcf_gin).fillna(0) "
    def interpolate_pcf(self, r, pcf_r, **params):
        """Interpolate the pair correlation function (pcf) from evaluations ``(r, pcf_r)``.

        Args:
            r: vector containing the radius on which the pair correlation function is evaluated.
            pcf_r: vector containing the evaluations of the pair correlation function on r_vec.
            params: dict of the parameters of :py:func:`scipy.interpolate.interp1d` function.

        # todo clarify whether is it F[g] or F[g-1]
        """

        return interpolate.interp1d(r, pcf_r, **params)

    def compute_structure_factor(self, k, pcf, method="Ogata", **params):
        r"""Compute the `structure factor <https://en.wikipedia.org/wiki/Radial_distribution_function#The_structure_factor>`_ of the underlying point process at ``k`` from its pair correlation function ``pcf`` (assumed to be radially symmetric).

        .. math::

            SF(k) = 1 + \rho F[g-1](k)

        where
        - :math:`\rho` is the intensity of the point process ``self.intensity``
        - :math:`g` is the corresponding radially symmetric pair correlation function ``pcf``. Note that :math:`g-1` is also called total pair correlation function.

        Args:
            k (np.ndarray): norm of the wave vectors where the structure factor is to be evaluated.
            pcf ([type]): callable radially symmetric pair correlation function :math:`g`.
            method (str, optional): select the method to compute the `Radially Symmetric Fourier transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` as a Hankel transform :py:class:`HankelTransFormOgata` or :py:class:`HankelTransFormBaddourChouinard`.
            Choose between "Ogata" or "BaddourChouinard". Defaults to "Ogata".
            params: parameters passed to the corresponding Hankel transform
            - ``method == "Ogata"``
                params = dict(step_size=..., nb_points=...)
            - ``method == "BaddourChouinard"``
                params = dict(
                    r_max=...,
                    nb_points=...,
                    interpolotation=dict(:py:func:`scipy.integrate.interp1d` parameters)
                )
        Returns:
            np.ndarray: :math:`SF(k)` evaluation of the structure factor at ``k``.

        .. important::

            The Fourier transform involved <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` is computed via

        .. note::

            Typical usage: ``pcf`` is estimated using :py:meth:`StructureFactor.compute_pcf` and then interpolated using :py:meth:`StructureFactor.interpolate_pair_correlation_function`.

        # todo clarify whether is it F[g] or F[g-1]
        # todo
        """
        assert callable(pcf)
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        ft_k = ft.transform(total_pcf, k, method=method, **params)
        return 1.0 + self.intensity * ft_k
