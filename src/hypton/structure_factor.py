import numpy as np
import scipy.interpolate as interpolate
import pandas as pd
import rpy2.robjects as robjects

import hypton.utils as utils
from hypton.point_pattern import PointPattern
from hypton.transforms import RadiallySymmetricFourierTransform
from hypton.spatial_windows import BoxWindow

from hypton.spatstat_interface import SpatstatInterface


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
    # ! Mettre un warning que scattering_intensity marche seulement dans les cubic windows, pcf pour dimension 2 et 3 seulement, hankel pour isotropic en dimension 2, en dimension 3 faire un MC pour approximer l'integral
    def __init__(self, point_pattern):
        r"""#todo write docsting
        Args:
            points: :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
            intensity(float): intensity of the underlying point process represented by `points`.
        # todo ajouter une methode pour approximer l'intensité pour des stationnaire ergodic si elle n'est pas provided
        """
        assert isinstance(point_pattern, PointPattern)
        assert point_pattern.dimension == 2
        self.point_pattern = point_pattern
        self.intensity = point_pattern.intensity
        self.norm_k_min = None

    @property
    def dimension(self):
        return self.point_pattern.dimension

    def compute_sf_scattering_intensity(
        self,
        max_k,
        meshgrid_size=None,
        max_add_k=1,
    ):

        # todo replace the link below to the link of our future paper.
        # todo ajouter la possibilité d'entré  plusieur echantillion
        # todo utuliser l'intensité et le volume au lieu de N dans la formule i.e. remplacer N pas intensité*volume de la fenetre
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
                This is represented inside k_vectors defined as,
                :math:
                    `k_vectors = (2 \pi n_vector) /L`

                where, ``n_vector`` is a vector of integer from 1 into max_n, and ``L`` is the length side of the cubic window that contains ``points``. see # todo put the link of our paper

        Args:
            L (int): side length of the cubic window that contains ``points``.

            max_k (int): maximum norm of ``k_vector``. The user can't chose the ``k_vector`` (defined above) since there's only a specific allowed values of ``k_vector`` used in the estimation of the structure factor by the scattering intensity, but the user can  specify in ``max_k`` the maximum norm of ``k_vector``.
            # todo clarify the description, k_vector exists only in the code not in the docstring, the argument name is not clear
            meshgrid_size (int): if the requested evaluation is on a meshgrid,  then ``meshgrid_size`` is the number of waves in each row of the meshgrid. Defaults to None.
            plot_param (str): "true" or "false", parameter to precise whether to show the plot of to hide it. Defaults to "true"
            plot_type (str): ("plot", "color_level" and "all"), specify the type of the plot to be shown. Defaults to "plot".
            bins_number (int): number of bins used by binning_function to find the mean of ``self.scattering_intensity`` over subintervals. For more details see the function ``binning_function`` in ``utils``. Defaults to 20.

        Returns:
            :math:`\left\lVert |\mathbf{k}| \right\rVert, SI(\mathbf{k})`, the norm of ``k_vector`` represented by ``norm_k_vector`` and the estimation of the scattering intensity ``si`` evaluated at ``k_vector``.
        """
        point_pattern = self.point_pattern
        assert isinstance(point_pattern.window, BoxWindow)
        L = np.abs(
            point_pattern.window.bounds[0, 0] - point_pattern.window.bounds[1, 0]
        )
        max_n = np.floor(max_k * L / (2 * np.pi))  # maximum of ``k_vector``
        if meshgrid_size is None:  # Add extra allowed values near zero
            n_vector = np.linspace(1, max_n, int(max_n))
            k_vector = 2 * np.pi * np.column_stack((n_vector, n_vector)) / L

            max_add_n = np.floor(max_add_k * L / (2 * np.pi))
            add_n_vector = np.linspace(1, np.int(max_add_n), np.int(max_add_n))
            X, Y = np.meshgrid(add_n_vector, add_n_vector)
            add_k_vector = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L
            print(add_k_vector.shape)
            print(k_vector.shape)
            k_vector = np.concatenate((add_k_vector, k_vector))
            print(k_vector.shape)
        else:
            step_size = int((2 * max_n + 1) / meshgrid_size)
            if meshgrid_size > (2 * max_n + 1):
                step_size = 1
                # todo raise warning : meshgrid_size should be less than the total allowed number of points
            n_vector = np.arange(-max_n, max_n, step_size)
            n_vector = n_vector[n_vector != 0]
            X, Y = np.meshgrid(n_vector, n_vector)
            k_vector = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L

        si = utils.compute_scattering_intensity(k_vector, self.point_pattern.points)
        norm_k_vector = np.linalg.norm(k_vector, axis=1)

        if meshgrid_size is not None:
            norm_k_vector = norm_k_vector.reshape(X.shape)
            si = si.reshape(X.shape)

        return norm_k_vector, si

    def plot_scattering_intensity(
        self,
        norm_k,
        si,
        plot_type="plot",
        axes=None,
        exact_sf=None,
        error_bar=False,
        file_name="",
        **binning_params
    ):
        if plot_type == "plot":
            return utils.plot_si_showcase(
                norm_k, si, axes, exact_sf, error_bar, file_name, **binning_params
            )
        elif plot_type == "imshow":
            return utils.plot_si_imshow(norm_k, si, axes, file_name)

        elif plot_type == "all":
            return utils.plot_si_all(
                self.point_pattern,
                norm_k,
                si,
                exact_sf,
                error_bar,
                file_name,
                **binning_params
            )
        else:
            raise ValueError(
                "plot_type must be chosen among ('all', 'plot', 'imshow')."
            )

    def compute_pcf(self, method="fv", install_spatstat=False, **params):

        """Estimate the pair correlation function (pcf) of ``self.point_pattern`` using spatstat ``spastat.core.pcf_ppp`` or ``spastat.core.pcf_fv`` functions according to ``method`` called with the corresponding parameters ``params``.

        radius: is the radius of the ball containing the points on which the pair correlation function will be approximated
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

        data = self.point_pattern.convert_to_spatstat_ppp()

        if method == "ppp":
            r = params.get("r", None)
            if r is not None and isinstance(r, np.ndarray):
                params["r"] = robjects.vectors.FloatVector(r)
            pcf = spatstat.core.pcf_ppp(data, **params)

        if method == "fv":
            params_Kest = params.get("Kest", dict())
            Kest_r = params_Kest.get("r", None)
            if Kest_r is not None and isinstance(Kest_r, np.ndarray):
                params_Kest["r"] = robjects.vectors.FloatVector(Kest_r)
            k_ripley = spatstat.core.Kest(data, **params_Kest)
            params_fv = params.get("fv", dict())
            pcf = spatstat.core.pcf_fv(k_ripley, **params_fv)

        return pd.DataFrame(np.array(pcf).T, columns=pcf.names)

    def plot_pcf(self, pcf_dataframe, exact_pcf=None, file_name="", **kwargs):
        # kwargs : parameter of pandas.DataFrame.plot.line https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html
        return utils.plot_pcf(pcf_dataframe, exact_pcf, file_name, **kwargs)

    def interpolate_pcf(self, r, pcf_r, clean=False, **params):
        """Interpolate the pair correlation function (pcf) from evaluations ``(r, pcf_r)``.
        Note : if ``pcf_r``contains "Inf", you can clean the ``pcf_r``using the method ``cleaning_data``.

        Args:
            r: vector containing the radius on which the pair correlation function is evaluated.
            pcf_r: vector containing the evaluations of the pair correlation function on ``r``.
            params: dict of the parameters of :py:func:`scipy.interpolate.interp1d` function.
            clean: f ``pcf_r``contains "Inf", you can clean the ``pcf_r``using the method ``cleaning_data`` by setting clean="true".

        """
        params.setdefault("fill_value", "extrapolate")
        params.setdefault("kind", "cubic")
        rmin = np.min(r)
        rmax = np.max(r)
        if clean:
            pcf_r = utils.cleaning_data(pcf_r)
        return dict(rmin=rmin, rmax=rmax), interpolate.interp1d(r, pcf_r, **params)

    def compute_sf_hankel_quadrature(self, pcf, norm_k=None, method="Ogata", **params):
        r"""Compute the `structure factor <https://en.wikipedia.org/wiki/Radial_distribution_function#The_structure_factor>`_ of the underlying point process at ``k`` from its pair correlation function ``pcf`` (assumed to be radially symmetric).

        .. math::

            S(\mathbf{k}) = 1 + \rho \mathcal{F}( g-1)(\mathbf{k})

        where
        - :math:`\rho` is the intensity of the point process ``self.intensity``
        - :math:`g` is the corresponding radially symmetric pair correlation function ``pcf``. Note that :math:`g-1` is also called total pair correlation function.

        Args:
            nom_k (np.ndarray): vector containing the norms of the waves where the structure factor is to be evaluated.
            pcf ([type]): callable radially symmetric pair correlation function :math:`g`.
            method (str, optional): select the method to compute the `Radially Symmetric Fourier transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` as a Hankel transform :py:class:`HankelTransFormOgata` or :py:class:`HankelTransFormBaddourChouinard`.
            Choose between "Ogata" or "BaddourChouinard". Defaults to "Ogata".
            params: parameters passed to the corresponding Hankel transform

            - ``method == "Ogata"``
                params = dict(step_size=..., nb_points=...)
            - ``method == "BaddourChouinard"``
                params = dict(
                    rmax=...,
                    nb_points=...,
                    interpolotation=dict(:py:func:`scipy.integrate.interp1d` parameters)
                )
        Returns:
            np.ndarray: :math:`SF(k)` evaluation of the structure factor at ``k``.

        .. important::

            The Fourier transform involved <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` is computed via
            # todo via what???
        .. note::

            Typical usage: ``pcf`` is estimated using :py:meth:`StructureFactor.compute_pcf` and then interpolated using :py:meth:`StructureFactor.interpolate_pair_correlation_function`.
        """
        assert callable(pcf)
        if method == "Ogata" and norm_k.all() is None:
            raise ValueError(
                "norm_k is not optional while using method='Ogata'. Please provide a vector norm_k in the input. "
            )
        params.setdefault("rmax", None)
        if method == "BaddourChouinard" and params["rmax"] is None:
            raise ValueError(
                "rmax is not optional while using method='BaddourChouinard'. Please provide rmax in the input. "
            )
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        norm_k, ft_k = ft.transform(total_pcf, norm_k, method=method, **params)
        if method == "Ogata" and params["rmax"] is not None:
            params.setdefault("step_size", 0.1)
            step_size = params["step_size"]
            self.norm_k_min = (2.7 * np.pi) / (params["rmax"] * step_size)
        return norm_k, 1.0 + self.intensity * ft_k

    def plot_sf_hankel_quadrature(
        self,
        norm_k,
        sf,
        axis=None,
        norm_k_min=None,
        exact_sf=None,
        error_bar=False,
        file_name="",
        **binning_params
    ):

        return utils.plot_sf_hankel_quadrature(
            norm_k,
            sf,
            axis,
            norm_k_min,
            exact_sf,
            error_bar,
            file_name,
            **binning_params
        )
