import numpy as np
import scipy.interpolate as interpolate
import pandas as pd
import rpy2.robjects as robjects

from structure_factor.utils import (
    compute_scattering_intensity,
    plot_scattering_intensity_,
    plot_pcf_,
    cleaning_data,
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
    # ! Mettre un warning que scattering_intensity marche seulement dans les cubic windows, pcf pour dimension 2 et 3 seulement, hankel pour isotropic en dimension 2, en dimension 3 faire un MC pour approximer l'integral
    def __init__(self, point_pattern):
        r"""
        Args:
            points: :math:`n \times 2` np.array representing a realization of a 2 dimensional point process.
            intensity(float): intensity of the underlying point process represented by `points`.
        # todo a distcuter les following todo avec Rémi le jeudi
        # todo treat the case where the intensity is not provided
        # todo consider passing the window where the points were observed to avoid radius in compute_pcf(self, radius...
        # todo ajouter en entré un parametre qui prend le window dans le quel les data sont obtenu et le passer à spatstat en pcf
        # todo ajouter une methode pour approximer l'intensité pour des stationnaire ergodic si elle n'est pas provided
        """
        dimension = point_pattern.points.shape[1]
        assert point_pattern.points.ndim == 2 and dimension == 2
        self.dimension = dimension
        self.point_pattern = point_pattern
        self.intensity = point_pattern.intensity

    def compute_scattering_intensity(
        self,
        maximum_k,
        meshgrid_size=None,
        max_add_n=10,
    ):
        # todo je peux à la place de max_add_n mettre max_k_add plus compréhensible et changer en bas comme pour k_vector
        # todo replace the link below to the link of our future paper.
        # todo fit a line to the binned si
        # todo ajouter des interval de confiance sur les binned values après faire un binning
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

                where, ``n_vector`` is a vector of integer from 1 into maximum_n, and ``L`` is the length side of the cubic window that contains ``points``. see # todo put the link of our paper

        Args:
            L (int): side length of the cubic window that contains ``points``.
            # todo Consider passing a PointPattern at initialization with .points and .window attributes
            maximum_k (int): maximum norm of ``k_vector``. The user can't chose the ``k_vector`` (defined above) since there's only a specific allowed values of ``k_vector`` used in the estimation of the structure factor by the scattering intensity, but the user can  specify in ``maximum_k`` the maximum norm of ``k_vector``.
            # todo clarify the description, k_vector exists only in the code not in the docstring, the argument name is not clear
            meshgrid_size (int): if the requested evaluation is on a meshgrid,  then ``meshgrid_size`` is the number of waves in each row of the meshgrid. Defaults to None.
            plot_param (str): "true" or "false", parameter to precise whether to show the plot of to hide it. Defaults to "true"
            plot_type (str): ("plot", "color_level" and "all"), specify the type of the plot to be shown. Defaults to "plot".
            bins_number (int): number of bins used by binning_function to find the mean of ``self.scattering_intensity`` over subintervals. For more details see the function ``binning_function`` in ``utils``. Defaults to 20.

        Returns:
            :math:`\left\lVert |\mathbf{k}| \right\rVert, SI(\mathbf{k})`, the norm of ``k_vector`` represented by ``norm_k_vector`` and the estimation of the scattering intensity ``si`` evaluated at ``k_vector``.
        """
        L = np.abs(
            point_pattern.window.bounds[0, 0] - point_pattern.window.bounds[1, 0]
        )
        maximum_n = np.floor(maximum_k * L / (2 * np.pi))  # maximum of ``k_vector``
        if meshgrid_size is None:
            n_vector = np.linspace(1, maximum_n, int(maximum_n))
            k_vector = 2 * np.pi * np.column_stack((n_vector, n_vector)) / L
            add_n_vector = np.linspace(1, np.int(max_add_n), np.int(max_add_n))
            add_n_grid, add_n_grid = np.meshgrid(add_n_vector, add_n_vector)
            add_k_vector = (
                2
                * np.pi
                * np.column_stack((add_n_grid.ravel(), add_n_grid.ravel()))
                / L  # adding allowed values near zero
            )
        else:
            n_vector = np.arange(1, maximum_n, int(maximum_n / meshgrid_size))
            print(n_vector.shape)
            print(n_vector)
            X, Y = np.meshgrid(n_vector, n_vector)
            k_vector = 2 * np.pi * np.column_stack((X.ravel(), Y.ravel())) / L

        si = compute_scattering_intensity(k_vector, self.point_pattern.points)
        norm_k_vector = np.linalg.norm(k_vector, axis=1)

        if meshgrid_size is None:
            add_si = compute_scattering_intensity(
                add_k_vector, self.point_pattern.points
            )
            norm_add_k_vector = np.linalg.norm(add_k_vector, axis=1)
            si = np.concatenate((add_si, si), axis=None)

            norm_k_vector = np.concatenate(
                (norm_add_k_vector, norm_k_vector), axis=None
            )

        if meshgrid_size is not None:

            norm_k_vector = norm_k_vector.reshape(
                X.shape
            )  # reshape the ``norm_k_vector`` to the correct shape
            si = si.reshape(
                X.shape
            )  # reshape the scattering intensity ``si`` to the correct shape

        return norm_k_vector, si

    # todo faire une fonction qui calcule les allowed values

    def plot_scattering_intensity(
        self, wave_length, si, plot_type="plot", exact_sf=None, **binning_params
    ):
        points = self.point_pattern.points
        return plot_scattering_intensity_(
            points, wave_length, si, plot_type, exact_sf, **binning_params
        )

    def compute_pcf(self, method="fv", install_spatstat=False, **params):
        # todo consider choosing a different window shape
        """Estimate the pair correlation function (pcf) of ``self.points`` observed in a disk window centered at the origin with radius ``radius`` using spatstat ``spastat.core.pcf_ppp`` or ``spastat.core.pcf_fv`` functions according to ``method`` called with the corresponding parameters ``params``.

        # todo consider adding the window where points were observed at __init__ to avoid radius argument.
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

    def plot_pcf(self, pcf_DataFrame, exact_pcf=None, **kwargs):
        # kwargs : parameter of pandas.DataFrame.plot.line https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html
        return plot_pcf_(pcf_DataFrame, exact_pcf, **kwargs)

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
        r_min = np.min(r)
        r_max = np.max(r)
        if clean:
            pcf_r = cleaning_data(pcf_r)
        return dict(r_min=r_min, r_max=r_max), interpolate.interp1d(r, pcf_r, **params)

    # todo à voir pourquoi ``r`` n'est pas en entrée pcf n'est pas tout le temps une fonction . to see in detail in the second check (pour Diala)
    def compute_structure_factor_via_hankel(
        self, pcf, k=None, method="Ogata", **params
    ):
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
            # todo à la place de faire une méthod d'interpolation puis passé la fonction intérpolé à "Ogata" on peut la faire à l'interieur de "Ogata" comme "BaddourChouinard". to see in detail in the second check Diala...
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
            # todo ``pcf`` could be a function ... to see in detail .... diala
            The Fourier transform involved <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_ of :math:`g` is computed via
            # todo via what???
        .. note::

            Typical usage: ``pcf`` is estimated using :py:meth:`StructureFactor.compute_pcf` and then interpolated using :py:meth:`StructureFactor.interpolate_pair_correlation_function`.
        """
        assert callable(pcf)
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        k_, ft_k = ft.transform(total_pcf, k, method=method, **params)
        if method == "Ogata" and r_max is not None:
            params.setdefault("step_size", 0.1)
            self.k_min = ft.compute_k_min(params["r_max"], params["step_size"])
        return k_, 1.0 + self.intensity * ft_k
