import numpy as np
import scipy.interpolate as interpolate
import pandas as pd
import rpy2.robjects as robjects

from structure_factor.utils import (
    compute_scattering_intensity,
    plot_scattering_intensity_estimate,
)
from structure_factor.transforms import RadiallySymmetricFourierTransform
from structure_factor.spatstat_interface import SpatstatInterface


class StructureFactor:
    r"""Implementation of various estimators of the structure factor of a 2 dimensional point process of intensity :math:`\rho`, defined as

    .. math::

        S(\mathbf{k}) = 1 + \rho \mathcal{F}(g-1)(\mathbf{k}),

    where :math:`\mathcal{F}` denotes the Fourier transform
          :math:`g` corresponds to the pair correlation function (also known as radial distribution function) # add link
          :math:`\mathbf{k}` is a wave in :math:`\mathhb{R}^2`

    .. seealso::

        page 8 http://chemlabs.princeton.edu/torquato/wp-content/uploads/sites/12/2018/06/paper-401.pdf
    """

    def __init__(self, points, intensity):
        r"""
        Args:
            points: :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
            intensity: intensity of the underlying point process represented by `points`.
        # todo a distcuter les following todo avec Rémi le jeudi
        # todo treat the case where the intensity is not provided
        # todo consider passing the window where the points were observed to avoid radius in compute_pcf(self, radius...
        # todo est ce qu'on fait une class pointPatern  qui contient le window et les data et on le coupe pour que scattering intensity marche?
        """
        dimension = points.shape[1]
        assert points.ndim == 2 and dimension == 2
        self.dimension = dimension
        self.points = points
        self.intensity = intensity

    def compute_scattering_intensity(
        self,
        L,
        maximum_wave,
        meshgrid_size=None,
        plot_param="true",
        bins_number=20,
        plot_type="plot",
    ):
        # todo replace the link below to the link of our future paper.
        # todo si should be binned then a line should be fitted representing si
        r"""Compute the ensemble estimator of the scattering intensity described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5).

        .. math::

            SI(k) =
            \left\lvert
                \sum_{x \in \mathcal{X}}
                    \exp(- i \left\langle k, x \right\rangle)
            \right\rvert^2

        Note: This estimation converges to the structure factor in the thermodynamic limits.

        Notes:  The points should be simulated inside a cube. # todo see L arg
                The allowed values of wave vectors are the points of the dual lattice of the lattice having fundamental cell the cubic window.
                This is represented inside wave_vectors defined as, math: `wave_vectors = (2 \pi k_vector) /L`, where k_vector is a vector of integer from 1 into maximum_k, and L in the length side of the cubic window that contains `points`. see # todo put the link of our paper

        Args:
            L (int): side length of the cubic window that contains ``points``.
            # todo What if the window is not cubic?
            # todo Consider passing a PointPattern at initialization with .points and .window attributes
            maximum_wave (int): maximum norm of ``wave_vector``. The user can't chose the ``wave_vector`` (defined above) since there's only a specific allowed values of ``wave_vector`` used in the estimation of the structure factor by the scattering intensity, but the user can  specify in ``maximum_wave`` the maximum norm of ``wave_vector``.
            # todo clarify the description, wave_vector exists only in the code not in the docstring, the argument name is not clear
            meshgrid_size (int): if the requested evaluation is on a meshgrid,  then ``meshgrid_size`` is the number of waves in each row of the meshgrid. Defaults to None.
            plot_param (str): "true" or "false", parameter to precise whether to show the plot of to hide it. Defaults to "true"
            plot_type (str): ("plot", "color_level" and "all"), specify the type of the plot to be shown. Defaults to "plot".
            bins_number (int): number of bins used by binning_function to find the mean of ``self.scattering_intensity`` over subintervals. For more details see the function ``binning_function`` in ``utils``. Defaults to 20.

        Returns:
            :math:`\left\lVert k \right\rVert, SI(K)`, the norm of the wave vectors :math:`k` and the estimation of the scattering intensity evaluated at :math:`k`.
        """
        maximum_k = np.floor(
            maximum_wave * L / (2 * np.pi * np.sqrt(2))
        )  # maximum of ``k_vector``
        if meshgrid_size is None:
            k_vector = np.linspace(1, maximum_k, int(maximum_k))
            wave_vector = 2 * np.pi * np.column_stack((k_vector, k_vector)) / L
        else:
            x_grid = np.linspace(-maximum_wave, maximum_wave, int(meshgrid_size))
            X, Y = np.meshgrid(x_grid, x_grid)
            wave_vector = np.column_stack((X.ravel(), Y.ravel()))

        si = compute_scattering_intensity(wave_vector, self.points)
        wave_length = np.linalg.norm(wave_vector, axis=1)

        if meshgrid_size is not None:
            wave_length = wave_length.reshape(
                X.shape
            )  # reshape the ``wave_vector`` to the correct shape
            si = si.reshape(
                X.shape
            )  # reshape the scattering intensity ``si`` to the correct shape
        if plot_param == "true":
            plot_scattering_intensity_estimate(wave_length, si, plot_type, bins_number)
        return wave_length, si

    def compute_pcf(self, radius, method, install_spatstat=False, **params):
        # todo consider choosing a different window shape
        """Estimate the pair correlation function (pcf) of ``self.points`` observed in a disk window centered at the origin with radius ``radius`` using spatstat ``spastat.core.pcf_ppp`` or ``spastat.core.pcf_fv`` functions according to ``method`` called with the corresponding parameters ``params``.

        # todo consider adding the window where points were observed at __init__ to avoid radius argument.
        radius: # todo expliciter le radius fais quoi
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

        window = spatstat.geom.disc(radius=radius)

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
    # ! no need to call pandas for this, have a look at np.nan_to_num https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html

    def interpolate_pcf(self, r, pcf_r, **params):
        """Interpolate the pair correlation function (pcf) from evaluations ``(r, pcf_r)``.
        Note : if ``pcf_r``contains "Inf", you can clean the ``pcf_r``using the method ``cleaning_data``.

        Args:
            r: vector containing the radius on which the pair correlation function is evaluated.
            pcf_r: vector containing the evaluations of the pair correlation function on ``r``.
            params: dict of the parameters of :py:func:`scipy.interpolate.interp1d` function.

        """
        return interpolate.interp1d(r, pcf_r, **params)

    # todo à voir pourquoi ``r`` n'est pas en entrée pcf n'est pas tout le temps une fonction . to see in detail in the second check
    def compute_structure_factor(self, k, pcf, method="Ogata", **params):
        r"""Compute the `structure factor <https://en.wikipedia.org/wiki/Radial_distribution_function#The_structure_factor>`_ of the underlying point process at ``k`` from its pair correlation function ``pcf`` (assumed to be radially symmetric).

        .. math::

            S(\mathbf{k}) = 1 + \rho \mathcal{F}( g-1)(\mathbf{k})

        where
        - :math:`\rho` is the intensity of the point process ``self.intensity``
        - :math:`g` is the corresponding radially symmetric pair correlation function ``pcf``. Note that :math:`g-1` is also called total pair correlation function.

        Args:
            k (np.ndarray): vector containing the norms of the waves where the structure factor is to be evaluated.
            pcf ([type]): callable radially symmetric pair correlation function :math:`g`.
            method (str, optional): select the method to compute the `Radially Symmetric Fourier transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` as a Hankel transform :py:class:`HankelTransFormOgata` or :py:class:`HankelTransFormBaddourChouinard`.
            Choose between "Ogata" or "BaddourChouinard". Defaults to "Ogata".
            params: parameters passed to the corresponding Hankel transform
            # todo à la place de faire une méthod d'interpolation puis passé la fonction intérpolé à "Ogata" on peut la faire à l'interieur de "Ogata" comme "BaddourChouinard". to see in detail in the second check...
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
            # todo ``pcf`` could be a function ... to see in detail ....
            The Fourier transform involved <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` is computed via

        .. note::

            Typical usage: ``pcf`` is estimated using :py:meth:`StructureFactor.compute_pcf` and then interpolated using :py:meth:`StructureFactor.interpolate_pair_correlation_function`.
        """
        assert callable(pcf)
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        ft_k = ft.transform(total_pcf, k, method=method, **params)
        return 1.0 + self.intensity * ft_k
