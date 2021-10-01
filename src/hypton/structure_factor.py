import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.interpolate as interpolate
import warnings

import hypton.utils as utils
from hypton.point_pattern import PointPattern
from hypton.spatial_windows import BoxWindow
from hypton.spatstat_interface import SpatstatInterface
from hypton.transforms import RadiallySymmetricFourierTransform


class StructureFactor:
    r"""Implementation of various estimators of the `structure factor <https://en.wikipedia.org/wiki/Structure_factor>`_ :math:`S` of a 2 dimensional stationary ergodic point process :math:`\mathcal{X}` with intensity :math:`\rho`, defined as

    .. math::

        S(\mathbf{k}) = 1 + \rho \mathcal{F}(g-1)(\mathbf{k}),

    where :math:`\mathcal{F}` denotes the Fourier transform, :math:`g` the pair correlation function corresponds to :math:`\mathcal{X}`, :math:`\mathbf{k} \in \mathbb{R}^2` is a wave vector and we denote the wave length :math:`k = \| \mathbf{k} \|_2`.

    .. seealso::

        :cite:`Tor18` page 8.
    """
    # ! Mettre un warning que scattering_intensity marche seulement dans les cubic windows, pcf pour dimension 2 et 3 seulement, hankel pour isotropic en dimension 2, en dimension 3 faire un MC pour approximer l'integral

    def __init__(self, point_pattern):
        r"""
        Args:
            point_pattern (:py:class:`~hypton.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.points`` and (optionally) the ``point_pattern.intensity`` of the point process.
        """
        assert isinstance(point_pattern, PointPattern)
        assert point_pattern.dimension == 2
        self.point_pattern = point_pattern
        self.intensity = point_pattern.intensity
        self.norm_k_min = None

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    def compute_sf_scattering_intensity(
        self,
        k_vector=None,
        max_k=None,
        meshgrid_size=None,
        max_add_k=1,
    ):
        r"""Compute the scattering intensity which is an ensemble estimator of the structure factor of an ergodic stationary point process :math:`\mathcal{X} \subset \mathbb{R}^2`, defined by

        .. math::

            SI(\mathbf{k}) =
            \left\lvert
                \sum_{x \in \mathcal{X}}
                    \exp(- i \left\langle \mathbf{k}, \mathbf{x} \right\rangle)
            \right\rvert^2

        where :math:`\mathbf{k} \in \mathbb{R}^2` is a wave vector.

        The scattering intensity converges to the structure factor in the thermodynamic limits.
        The scattering intensity can be evaluated on any vector (np.array or meshgrid) of waves by precising the argument k_vector.
        Nevertheless, the estimation of the structure factor by the scattering intensity is valid for point process sampled in a cubic window (or restricted to a box window via the method restrict_to_window of the class :py:class:`~hypton.point_pattern.PointPattern` for more details see paper...) and on a specific vector of allowed values of waves corresponding to the dual of the lattice having as fundamental cell the sample of points.
        In other words, if the points are simulated in a cubic window of length :math:`L`, then the vector of allowed is

        .. math::
            \{
            \frac{2 \pi}{L} \mathbf{n},\,
            \text{for} \; \mathbf{n} \in (\mathbb{Z}^d)^\ast \}

        So it's recommended to not specify the vector of wave ``k_vector``, but to either specify a meshgrid size and the maximum component of the wave vector respectively via ``meshgrid_size`` and ``max_k`` if you need to evaluate the scattering intensity on a meshgrid of allowed values (see example of ...) or just the maximum component of the wave vector ``max_k`` if you need to evaluate the scattering intensity on a vector of allowed values. see :py:meth:`utils.allowed__wave_values`.

        Args:

            k_vector (list): list containing the 2 numpy.ndarray corresponding to the x and y components of the wave vector. As we mentioned before it recommended to keep the default k_vector and to specify max_k instead, so that the approximation will be evaluated on a list of allowed values. Defaults to None.

            max_k (float, optional): Maximum component of the allowed wave vector. Defaults to None.

            meshgrid_size (int, optional): Size of the meshgrid of allowed values if ``k_vector`` is set to None and ``max_k`` is specified. Warning: setting big value in ``meshgrid_size`` could be time consuming and harmful to your machine for large sample of points. Defaults to None.

            max_add_k (int, optional): Maximum component of the allowed wave vectors to be add. In other words, in the case of the evaluation on a vector of allowed values (without specifying ``meshgrid_size``),  ``max_add_k`` can be used to add allowed values in a certain region for better precision. Warning: setting big value in ``max_add_k`` could be time consuming and harmful to your machine for large sample of points. Defaults to 1.

        Returns:
            norm_k_vector (numpy.ndarray): The vector of wave length (i.e. the vector of norms of the wave vectors) on which the scattering intensity is evaluated.
            si (numpy.ndarray): The evaluation of the scattering intensity corresponding to the vector of wave length ``norm_k_vector``.
        """
        # todo ajouter la possibilité d'entré  plusieur echantillion
        # todo utuliser l'intensité et le volume au lieu de N dans la formule i.e. remplacer N pas intensité*volume de la fenetre
        if self.dimension != 2:
            warnings.warn(
                message="The package actually is applicable for 2 dimensional point process",
                category=DeprecationWarning,
            )

        point_pattern = self.point_pattern

        if not isinstance(point_pattern.window, BoxWindow):
            warnings.warn(
                message="The window should be a BoxWindow for that the scattering intensity consists an approximation of the structure factor. Hint: use PointPattern.restrict_to_window."
            )
        if k_vector is None:
            assert isinstance(point_pattern.window, BoxWindow)
            L = np.abs(
                point_pattern.window.bounds[0, 0] - point_pattern.window.bounds[1, 0]
            )
            k_vector = utils.allowed_wave_values(
                L=L, max_k=max_k, meshgrid_size=meshgrid_size, max_add_k=max_add_k
            )

        else:
            shape_x_k_vector = k_vector[0].shape
            k_vector = np.column_stack((k_vector[0].ravel(), k_vector[1].ravel()))
        si = utils.compute_scattering_intensity(k_vector, point_pattern.points)
        norm_k_vector = np.linalg.norm(k_vector, axis=1)

        if meshgrid_size is not None or len(shape_x_k_vector) == 2:
            shape_mesh = int(np.sqrt(norm_k_vector.shape[0]))
            norm_k_vector = norm_k_vector.reshape(shape_mesh, shape_mesh)
            si = si.reshape(shape_mesh, shape_mesh)

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
        window_res=None,
        **binning_params
    ):
        """Plot the result of the method :py:meth:`compute_sf_scattering_intensity`.
        You can add the theoretical structure factor using ``exact_sf`` and visualize the mean and the variance over bins of the scattering intensity by specifying ``error_bar=True`` (this is donne using a binning method :py:meth:`utils._bin_statistics`).
        The figure could be saved by specifying ``file_name``.

        Args:
            norm_k (numpy.ndarray): vactor of wave length.

            si (numpy.ndarray): vector of scattering intensity.

            plot_type (str, optional): ("plot", "imshow", "all"). Precision of the type of the plot we want to visualize. If "plot" is used then the output is a loglog plot. If "imshow" is used then the output is a color level plot. if "all" is used the the results are 3 subplots: the point pattern (or a restriction to a specific window if ``window_res`` is set), the loglog plots, and the color level plot. Note that you can not use the option "imshow" or "all", if ``norm_k`` is not a meshgrid. Defaults to "plot".

            axes (axis, optional): the support axis of the plots. Defaults to None.

            exact_sf (callable, optional): function representing the theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): Defaults to False. When set to ``True``, ``norm_k`` is divided into bins and the mean and the standard deviation over each bin are derived and visualized on the plot. Note that the error bar represent 3 times the standard deviation.

            file_name (str, optional): name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

            window_res (:py:class:`~hypton.spatial_windows.AbstractSpatialWindow`, optional): This could be used when the sample of points is large, so for time and visualization purpose it's better to restrict the plot of the sample of points to a smaller window.  Defaults to None.
        """

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
                window_res,
                **binning_params
            )
        else:
            raise ValueError(
                "plot_type must be chosen among ('all', 'plot', 'imshow')."
            )

    def compute_pcf(self, method="fv", install_spatstat=False, **params):
        """Estimate the pair correlation function of an **isotropic** point process process :math:`\mathcal{X} \subset \mathbb{R}^2`. The two methods that can be used are the methods ``spastat.core.pcf_ppp`` and ``spastat.core.pcf_fv`` of the the `R` package `spatstat <https://github.com/spatstat/spatstat>`_.

        .. warning::

            This function relies on the `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_ Python package which requires the `R programming language <https://cran.r-project.org/>`_ to be installed, but experience with R it is not required.

        Args:
            method (str, optional): Defaults to ``"fv"``. Choose between ``"ppp"`` or ``"fv"`` referring respectively to `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ and `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_ functions. These 2 methods approximate the pair correlation function of a point process from a realization of the underlying point process using some edge corrections and some basic approximations. For more details `see <https://www.routledge.com/Spatial-Point-Patterns-Methodology-and-Applications-with-R/Baddeley-Rubak-Turner/p/book/9781482210200>`_.

            install_spatstat (bool, optional): If it is set to ``True`` then the `R` package `spatstat <https://github.com/spatstat/spatstat>`_  will be updated or installed (if not present), see also the `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_ Python package.

            Keyword Args (params):

                - if ``method = "ppp"``
                    - keyword arguments of `spastat.core.pcf.ppp <https://rdrr.io/cran/spatstat.core/man/pcf.ppp.html>`_)

                - if ``method = "fv"``
                    - Kest = dict(keyword arguments of `spastat.core.Kest <https://rdrr.io/github/spatstat/spatstat.core/man/Kest.html>`_),
                    - fv = dict( keyword arguments of `spastat.core.pcf.fv <https://rdrr.io/cran/spatstat.core/man/pcf.fv.html>`_)

        Returns:
            pandas.DataFrame: version of the output of `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ of `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_.
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
        """Display the data frame output from the method :py:meth:`compute_pcf`.

        Args:
            pcf_dataframe (pandas.DataFrame): output DataFrame of the method :py:meth:`compute_pcf`.

            exact_pcf (callable, optional): function representing the theoretical pair correlation function of the point process. Defaults to None.

            file_name (str, optional): name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        Keyword Args (kwargs):
            Keyword arguments of the function `pandas.DataFrame.plot.line <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html>`_.
        """
        return utils.plot_pcf(pcf_dataframe, exact_pcf, file_name, **kwargs)

    def interpolate_pcf(self, r, pcf_r, clean=True, **params):
        """Interpolate the pair correlation function (pcf) evaluated at ``r``.

        Args:
            r (numpy.ndarray): vector of radius.

            pcf_r (numpy.ndarray): vector of approximation of the pair correlation function.

            clean (bool, optional): replace nan, posinf, neginf values to ``pcf_r`` by zeros before interpolation. Defaults to True.

        Keyword Args (params):
            Keyword arguments of the function `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

        Returns:
            tuple (dict, callable): dictionary containing the bounds of the interval containing the values of the vector ``r`` and the interpolated version of the pair correlation function.
        """
        params.setdefault("fill_value", "extrapolate")
        params.setdefault("kind", "cubic")
        rmin = np.min(r)
        rmax = np.max(r)
        if clean:
            pcf_r = utils.set_nan_inf_to_zero(pcf_r)
        return dict(rmin=rmin, rmax=rmax), interpolate.interp1d(r, pcf_r, **params)

    def compute_sf_hankel_quadrature(self, pcf, norm_k=None, method="Ogata", **params):
        r"""Compute the structure factor :math:`S` of the underlying **stationary isotropic** point process :math:`\mathcal{X} \subset \mathbb{R}^2`.
        This is done by first computing the Fourier transform of the pair correlation function :math:`g`, involving the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_, which is in turn computed via the quadrature shemes of :cite:`Oga05` or :cite:`BaCh15`.

        Args:
            pcf (callable): radially symmetric pair correlation function :math:`g`. You can get a discrete vector of estimation of the pair correlation function using the method :py:meth:`compute_pcf`, then interpolate the resulting vector using :py:meth:`interpolate_pcf` and pass the resulting function to the argument ``pcf``.

            # todo why not simply k ?
            norm_k (numpy.ndarray, optional): vector of wave lengths (i.e. norm of wave vectors) where the structure factor is to be evaluated. Defaults to None.

            method (str, optional): Choose between ``"Ogata"`` or ``"BaddourChouinard"``. Defaults to ``"Ogata"``. This selects the method used to compute the Fourier transform of :math:`g`, via the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_, see :py:class:`~hypton.transforms.HankelTransformOgata` and :py:class:`~hypton.transforms.HankelTransformBaddourChouinard`.

        Keyword Args (params):
            Keyword arguments passed to the corresponding Hankel transformer selected according to the ``method`` argument.

            - ``method == "Ogata"``, see :py:meth:`~hypton.transforms.HankelTransformOgata.compute_transformation_parameters`
                - ``step_size``
                - ``nb_points``

            - ``method == "BaddourChouinard"``, see :py:meth:`~hypton.transforms.HankelTransformBaddourChouinard.compute_transformation_parameters`
                - ``rmax``
                - ``nb_points``
                - ``interpolotation`` dictonnary containing the keyword arguments of `scipy.integrate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ parameters.

        Returns:
            tuple (np.ndarray, np.ndarray): vector of wave lengths ``k`` and the corresponding evaluation of the structure factor ``S(k)``.

        .. note::

            Typical usage: ``pcf`` is estimated using :py:meth:`compute_pcf` and then interpolated using :py:meth:`interpolate_pcf`.
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
            # todo il y a une fonct qui le fait why not used????
            self.norm_k_min = (2.7 * np.pi) / (params["rmax"] * step_size)
        sf = 1.0 + self.intensity * ft_k
        return norm_k, sf

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
        """Display the output of :py:meth:`compute_sf_hankel_quadrature`.
        You can add the theoretical structure factor using ``exact_sf`` (if it is known) and visualize the mean and the variance over bins of the scattering intensity by specifying ``error_bar=True`` (this is donne using a binning method :py:meth:`utils._bin_statistics`).
        The figure could be saved by specifying ``file_name``.

        Args:
            norm_k (numpy.1darray): vector of wave lengths (i.e. norms of waves) on which the structure factor is approximated.
            sf ([type]): [description]

            axis (axis, optional): the support axis of the plots. Defaults to None.

            norm_k_min (float, optional): estimation of an upper bounds for the allowed wave lengths. Defaults to None.

            exact_sf (callable, optional): function representing the theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): if it is set to ``True`` then, the ``norm_k`` is divided into bins and the mean and the standard deviation over each bin are derived and visualized on the plot. Note that the error bar represent 3 times the standard deviation. Defaults to False.

            file_name (str, optional): Defaults to "". Name used to save the figure. The available output formats depend on the backend being used.

        Returns:
            plot of the approximated structure factor.
        """

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
