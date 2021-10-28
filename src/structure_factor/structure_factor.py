import warnings

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.interpolate as interpolate
from spatstat_interface.interface import SpatstatInterface

import structure_factor.utils as utils
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BoxWindow, check_cubic_window
from structure_factor.transforms import RadiallySymmetricFourierTransform


class StructureFactor:
    r"""Implementation of various estimators of the structure factor :math:`S` of a d dimensional stationary ergodic point process :math:`\mathcal{X}` with intensity :math:`\rho`, as defined below.

    .. math::

        S(\mathbf{k}) = 1 + \rho \mathcal{F}(g-1)(\mathbf{k}),

    where :math:`\mathcal{F}` denotes the Fourier transform, :math:`g` the pair correlation function corresponds to :math:`\mathcal{X}`, :math:`\mathbf{k} \in \mathbb{R}^d` is a wave vector and we denote the wave length :math:`k = \| \mathbf{k} \|_2`.

    This class contains
        - Three estimators of the structure factor:
            - The scattering intensity :meth:`scattering_intensity`.
            - Estimator using Ogata quadrature for approximating the Hankel transform  :meth:`hankel_quadrature` with `method="Ogata"` :cite:`Oga05`.
            - Estimator using Baddour and Chouinard Discrete Hankel transform :meth:`hankel_quadrature` with `method="BaddourChouinard"` :cite:`BaCh15`.
        - Two estimators of the pair correlation function :
            - Estimator using Epanechnikov kernel and a bandwidth selected by Stoyan's rule of thumb :meth:`compute_pcf` with `method="ppp"`.
            - Estimator using the derivative of Ripley's K function :meth:`compute_pcf` with `method="fv"`.

            This 2 estimators are obtained using `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_ which builds a hidden interface with the package `spatstat <https://github.com/spatstat/spatstat>`_ of the programming language R.
        - An interpolation function :meth:`interpolate_pcf`, used to interpolate the result of :meth:`compute_pcf`.
        - Three plot methods :meth:`plot_scattering_intensity`,  :meth:`plot_pcf` and :meth:`plot_sf_hankel_quadrature` used to visualized the result of :meth:`scattering_intensity`, :meth:`compute_pcf` and :meth:`hankel_quadrature` respectively.


    .. seealso::

        :cite:`Tor18`, Section 2.1, equation (13).
    """

    # ! Mettre un warning que scattering_intensity marche seulement dans les cubic windows, pcf pour dimension 2 et 3 seulement, hankel pour isotropic en dimension 2, en dimension 3 faire un MC pour approximer l'integral

    def __init__(self, point_pattern):
        r"""Initialize StructureFactor from ``point_pattern``.

        Args:
            point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.

        """
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = point_pattern
        self.k_norm_min = None
        self.K_shape = None

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    def scattering_intensity(
        self,
        k=None,
        k_max=5,
        meshgrid_shape=None,
    ):
        r"""Compute the scattering intensity :math:`\widehat{S}_{SI}` which is an ensemble estimator of the structure factor :math:`S` of an ergodic stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d`, from a realization :math:`\mathcal{X}\cap W =\{x_i\}_{i=1}^N` of :math:`\mathcal{X}` within a **cubic** window :math:`W=[-L/2, L/2]^d`.

        .. math::

            \widehat{S}_{SI}(\mathbf{k}) =
             \frac{1}{N}\left\lvert
                \sum_{j=1}^N
                    \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
            \right\rvert^2

        for a specific sef of wavevectors

        .. math::
            \mathbf{k} \in \{
            \frac{2 \pi}{L} \mathbf{n},\,
            \text{for} \; \mathbf{n} \in (\mathbb{Z}^d)^\ast \}

        called in the physics literature **allowed values** or dual lattice :cite:`KlaLasYog20`.


        As the estimation of the structure factor :math:`S` via the scattering intensity :math:`\widehat{S}_{SI}` is valid for point processes sampled in a **cubic window**  and on a specific set of allowed wavevectors, so
            - If the sample :math:`\{x_j\}_{j=1}^N` does note lies in a cubic window, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample within a cubic window before using :meth:`scattering_intensity`.
            - :meth:`scattering_intensity` evalute the scattering intensity by default on the corresponding set of allowed wavevectors. But you can specify another set of wavevector by precising the argument ``k``.


        So it's recommended to not specify the vector of waves ``k``, but to either specify a meshgrid size and the maximum component of the set of wavevectors respectively via ``meshgrid_shape`` and ``k_max``, or just ``k_max``.

        .. note::

            Specifying the meshgrid size argument ``meshgrid_shape`` is usefull if the number of points of the realization is big so that in this case the evaluation of :math:`\widehat{S}_{SI}` on all the allowed wavevectors may be time consuming.

        .. seealso::

            :py:meth:`~structure_factor.utils.allowed_wave_vectors`.

        Args:

            k (np.ndarray): np.ndarray of d columns (where d is the dimesion of the space containing the points) where each row correspond to a wave vector. As mentioned before its recommended to keep the default ``k`` and to specify ``k_max`` instead, so that the approximation will be evaluated on allowed wavevectors. Defaults to None.

            k_max (float, optional): maximum component of the waves vectors i.e., for any output allowed wave vector :math:`\mathbf{k}=(k_1,...,k_d)`, we have :math:`k_i \leq k\_max` for all i. This implies that the maximum wave vectors will be :math:`(k\_max, ... k\_max)`. Defaults to 5.

            meshgrid_shape (tuple, optional): tuple of length `d`, where each element specify the number of component over the corresponding axis. It consists of the associated size of the meshgrid of allowed waves. For example if we are working in 2 dimensions, letting meshgid_shape=(2,3) will give a meshgrid of allowed waves formed by a vector of 2 values over the x-axis and a vectors of 3 values over the y-axis. Defaults to None.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k_norm: The vector of wavelengths (i.e. the vector of norms of the wave vectors) on which the scattering intensity was evaluated.
                - si: The evaluations of the scattering intensity corresponding to the vector of wave length ``k_norm``.

        Example:

            .. literalinclude:: code/si_example.py
                :language: python
                :lines: 1-21
                :emphasize-lines: 19-21
        """

        point_pattern = self.point_pattern
        window = point_pattern.window
        d = point_pattern.dimension
        # todo add assert k_max in a number

        if not isinstance(window, BoxWindow):
            warnings.warn(
                message="The window should be a 'cubic' BoxWindow for that the scattering intensity consists an approximation of the structure factor. Hint: use PointPattern.restrict_to_window."
            )
        if k is None:
            if meshgrid_shape is not None and len(meshgrid_shape) != d:
                raise ValueError(
                    "Each wave vector should belong to the same dimension (d) of the point process, i.e., len(meshgrid_shape) = d."
                )

            check_cubic_window(window)
            L = np.diff(window.bounds[0])

            k, K = utils.allowed_wave_vectors(
                d, L=L, k_max=k_max, meshgrid_shape=meshgrid_shape
            )
            self.K_shape = K[0].shape
        else:
            if k.shape[1] != d:
                raise ValueError(
                    "the vector of wave should belongs to the same dimension as the point process, i.e. k should have d columns"
                )
        si = utils.compute_scattering_intensity(k, point_pattern.points)
        k_norm = np.linalg.norm(k, axis=1)

        return k_norm, si

    def plot_scattering_intensity(
        self,
        k_norm,
        si,
        plot_type="radial",
        axes=None,
        exact_sf=None,
        error_bar=False,
        file_name="",
        window_res=None,
        **binning_params
    ):
        """Plot the result of the method :py:meth:`scattering_intensity`.

        You can add the theoretical structure factor using ``exact_sf`` and visualize the mean and the variance over bins of the scattering intensity by specifying ``error_bar=True`` (this is donne using a binning method :py:meth:`~structure_factor.utils._bin_statistics`).
        The figure could be saved by specifying ``file_name``.

        Args:
            k_norm (numpy.ndarray): vector of norms of the wavevectors .

            si (numpy.ndarray): approximated scattering intensity vector associted to `k_norm`.

            plot_type (str, optional): ("radial", "imshow", "all"). Type of the plot to visualize. If "radial", then the output is a loglog plot. If "imshow" (option available only for 2D point process), then the output is a color level 2D plot. If "all" (option available only for 2D point process), the results are 3 subplots: the point pattern (or a restriction to a specific window if ``window_res`` is set), the loglog radial plot, and the color level 2D plot . Note that the options "imshow" and "all" couldn't be used, if ``k_norm`` is not a meshgrid. Defaults to "radial".

            axes (axis, optional): the support axis of the plots. Defaults to None.

            exact_sf (callable, optional): function representing the theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): Defaults to False. When set to ``True``, ``k_norm`` is divided into bins and the mean and the standard deviation over each bin are derived and visualized on the plot. Note that the error bar represent the means +/- 3 standard deviation.

            file_name (str, optional): name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

            window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): This could be used when the sample of points is large, so for time and visualization purpose it's better to restrict the plot of the sample of points to a smaller window.  Defaults to None.

        Returns:
            plot of the approximated structure factor.

        Example:

            .. literalinclude:: code/si_example.py
                :language: python
                :lines: 23-29

            .. plot:: code/si_example.py
                :include-source: False
                :alt: alternate text
                :align: center
        """
        if plot_type == "radial":
            return utils.plot_si_showcase(
                k_norm, si, axes, exact_sf, error_bar, file_name, **binning_params
            )
        elif plot_type == "imshow":
            if self.dimension != 2:
                raise ValueError(
                    "This plot option is adpted only for 2D point process. Please use plot_type ='radial'."
                )
            if self.K_shape is None:
                raise ValueError(
                    "imshow require a meshgrid data. Choose plot_type= 'radial' or re-evaluate the scattering intensity on the meshgrid of allowed wave vectors."
                )
            si = si.reshape(self.K_shape)
            k_norm = k_norm.reshape(self.K_shape)
            return utils.plot_si_imshow(k_norm, si, axes, file_name)

        elif plot_type == "all":
            if self.dimension != 2:
                raise ValueError(
                    "This plot option is adpted only for 2D point process. Please use plot_type ='radial'."
                )
            if self.K_shape is None:
                raise ValueError(
                    "imshow require a meshgrid data. Choose plot_type ='radial' or re-evaluate the scattering intensity on the meshgrid of allowed wave vectors."
                )
            si = si.reshape(self.K_shape)
            k_norm = k_norm.reshape(self.K_shape)
            return utils.plot_si_all(
                self.point_pattern,
                k_norm,
                si,
                exact_sf,
                error_bar,
                file_name,
                window_res,
                **binning_params
            )
        else:
            raise ValueError(
                "plot_type must be chosen among ('all', 'radial', 'imshow')."
            )

    def compute_pcf(self, method="fv", install_spatstat=False, **params):
        r"""Estimate the pair correlation function of an **isotropic** point process process :math:`\mathcal{X} \subset \mathbb{R}^2`. The two methods that can be used are the methods ``spastat.core.pcf_ppp`` and ``spastat.core.pcf_fv`` of the the `R` package `spatstat <https://github.com/spatstat/spatstat>`_.

        .. warning::

            This function relies on the `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_ Python package which requires the `R programming language <https://cran.r-project.org/>`_ to be installed, but experience with R it is not required.

        Args:
            method (str, optional): Defaults to ``"fv"``. Choose between ``"ppp"`` or ``"fv"`` referring respectively to `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ and `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_ functions. These 2 methods approximate the pair correlation function of a point process from a realization of the underlying point process using some edge corrections and some basic approximations. For more details `see <https://www.routledge.com/Spatial-Point-Patterns-Methodology-and-Applications-with-R/Baddeley-Rubak-Turner/p/book/9781482210200>`_.

            install_spatstat (bool, optional): If it is set to ``True`` then the `R` package `spatstat <https://github.com/spatstat/spatstat>`_  will be updated or installed (if not present), see also the `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_ Python package.

        Keyword Args:

            params (dict):

                - if ``method = "ppp"``
                    - keyword arguments of `spastat.core.pcf.ppp <https://rdrr.io/cran/spatstat.core/man/pcf.ppp.html>`_)

                - if ``method = "fv"``
                    - Kest = dict(keyword arguments of `spastat.core.Kest <https://rdrr.io/github/spatstat/spatstat.core/man/Kest.html>`_),
                    - fv = dict( keyword arguments of `spastat.core.pcf.fv <https://rdrr.io/cran/spatstat.core/man/pcf.fv.html>`_)

        Returns:
            pandas.DataFrame: version of the output of `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ of `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_.

        Example:

            .. literalinclude:: code/pcf_example.py
                :language: python
                :lines: 1-15
                :emphasize-lines: 12-15
        """
        assert self.point_pattern.dimension == 2 or self.point_pattern.dimension == 3

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
        return pd.DataFrame(np.array(pcf).T, columns=pcf.names).drop(["theo"], axis=1)

    def plot_pcf(self, pcf_dataframe, exact_pcf=None, file_name="", **kwargs):
        """Display the data frame output from the method :py:meth:`compute_pcf`.

        Args:
            pcf_dataframe (pandas.DataFrame): output DataFrame of the method :py:meth:`compute_pcf`.

            exact_pcf (callable, optional): function representing the theoretical pair correlation function of the point process. Defaults to None.

            file_name (str, optional): name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        Keyword Args:

            kwargs (dict):

                Keyword arguments of the function `pandas.DataFrame.plot.line <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html>`_.

        Returns:
            plot of the approximated pair correlation function.

        Example:

            .. literalinclude:: code/pcf_example.py
                :language: python
                :lines: 16-20


            .. plot:: code/pcf_example.py
                :include-source: False
                :alt: alternate text
                :align: center
        """
        return utils.plot_pcf(pcf_dataframe, exact_pcf, file_name, **kwargs)

    def interpolate_pcf(self, r, pcf_r, clean=True, **params):
        """Interpolate the pair correlation function (pcf) evaluated at ``r``.

        Args:
            r (numpy.ndarray): vector of radius.

            pcf_r (numpy.ndarray): vector of approximation of the pair correlation function.

            clean (bool, optional): replace nan, posinf, neginf values to ``pcf_r`` by zeros before interpolation. Defaults to True.

        Keyword Args:

            params (dict):

                Keyword arguments of the function `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

        Returns:
            tuple (dict, callable): dictionary containing the bounds of the interval containing the values of the vector ``r`` and the interpolated version of the pair correlation function.

        Example:

            .. literalinclude:: code/sf_baddour_example.py
                :language: python
                :lines: 16-20

        """
        params.setdefault("fill_value", "extrapolate")
        params.setdefault("kind", "cubic")
        rmin = np.min(r)
        rmax = np.max(r)
        if clean:
            pcf_r = utils.set_nan_inf_to_zero(pcf_r)

        dict_rmin_rmax = dict(rmin=rmin, rmax=rmax)
        pcf = interpolate.interp1d(r, pcf_r, **params)
        return dict_rmin_rmax, pcf

    def hankel_quadrature(self, pcf, k_norm=None, method="Ogata", **params):
        r"""Compute the structure factor :math:`S` of the underlying **stationary isotropic** point process :math:`\mathcal{X} \subset \mathbb{R}^d`, which could be defined via the Hankel transform :math:`\mathcal{H}_{d/2 -1}` of order :math:`d/2 -1` as follow,

        .. math::

            S(\|\mathbf{k}\|) = 1 + \rho \frac{(2 \pi)^{d/2}}{\|\mathbf{k}\|^{d/2 -1}} \mathcal{H}_{d/2 -1}(\tilde g -1)(\|\mathbf{k}\|), \quad \tilde g:x \mapsto  g(x)x^{d/2 -1}.


        This method estimate the structure factor by approximating the corresponding Hankel transform via Ogata quadrature shemes of :cite:`Oga05` or Baddour and Chouinard Descrete Hankel transform :cite:`BaCh15`.

        .. warning::

            This method is actually applicable for 2 dimensional point process.

        Args:
            pcf (callable): radially symmetric pair correlation function :math:`g`. You can get a discrete vector of estimation of the pair correlation function using the method :py:meth:`compute_pcf`, then interpolate the resulting vector using :py:meth:`interpolate_pcf` and pass the resulting function to the argument ``pcf``.

            # todo why not simply k ?
            k_norm (numpy.ndarray, optional): vector of wave lengths (i.e. norm of wave vectors) where the structure factor is to be evaluated. Defaults to None.

            method (str, optional): Choose between ``"Ogata"`` or ``"BaddourChouinard"``. Defaults to ``"Ogata"``. This selects the method used to compute the Fourier transform of :math:`g`, via the `correspondence with the Hankel transform <https://en.wikipedia.org/wiki/Hankel_transform#Fourier_transform_in_d_dimensions_(radially_symmetric_case)>`_, see :py:class:`~structure_factor.transforms.HankelTransformOgata` and :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`.

        Keyword Args:

            params (dict):

                Keyword arguments passed to the corresponding Hankel transformer selected according to the ``method`` argument.

                - ``method == "Ogata"``, see :py:meth:`~structure_factor.transforms.HankelTransformOgata.compute_transformation_parameters`
                    - ``step_size``
                    - ``nb_points``

                - ``method == "BaddourChouinard"``, see :py:meth:`~structure_factor.transforms.HankelTransformBaddourChouinard.compute_transformation_parameters`
                    - ``rmax``
                    - ``nb_points``
                    - ``interpolotation`` dictonnary containing the keyword arguments of `scipy.integrate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ parameters.

        Returns:
            tuple (np.ndarray, np.ndarray):
                - k_norm: vector of wave lengths.
                - sf: the corresponding evaluation of the structure factor ``S(k)``.

        .. note::

            Typical usage: ``pcf`` is first estimated using :py:meth:`compute_pcf` then interpolated using :py:meth:`interpolate_pcf`, and the resulting function is used as input of this method.

        Example:

            .. literalinclude:: code/sf_baddour_example.py
                :lines: 1-29
                :emphasize-lines: 21-28

        """
        if self.dimension != 2:
            warnings.warn(
                message="This method is actually applicable for 2 dimensional point process",
                category=DeprecationWarning,
            )
        assert callable(pcf)
        if method == "Ogata" and k_norm.all() is None:
            raise ValueError(
                "k_norm is not optional while using method='Ogata'. Please provide a vector k_norm in the input. "
            )
        params.setdefault("rmax", None)
        if method == "BaddourChouinard" and params["rmax"] is None:
            raise ValueError(
                "rmax is not optional while using method='BaddourChouinard'. Please provide rmax in the input. "
            )
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        k_norm, ft_k = ft.transform(total_pcf, k_norm, method=method, **params)
        if method == "Ogata" and params["rmax"] is not None:
            params.setdefault("step_size", 0.1)
            step_size = params["step_size"]
            # todo il y a une fonct qui le fait why not used????
            self.k_norm_min = (2.7 * np.pi) / (params["rmax"] * step_size)
        sf = 1.0 + self.point_pattern.intensity * ft_k
        return k_norm, sf

    def plot_sf_hankel_quadrature(
        self,
        k_norm,
        sf,
        axis=None,
        k_norm_min=None,
        exact_sf=None,
        error_bar=False,
        file_name="",
        **binning_params
    ):
        """Display the output of :py:meth:`hankel_quadrature`.

        Pass the theoretical structure factor function through ``exact_sf`` (if it is known) and visualize the mean and the variance over bins of the scattering intensity by specifying ``error_bar=True`` (this is donne using a binning method :py:meth:`~structure_factor.utils._bin_statistics`). The figure can at ``file_name``.

        Args:
            k_norm (np.ndarray): vector of wave lengths (i.e. norms of waves) on which the structure factor is approximated.
            sf ([type]): [description]

            axis (axis, optional): the support axis of the plots. Defaults to None.

            k_norm_min (float, optional): estimation of an upper bounds for the allowed wave lengths. Defaults to None.

            exact_sf (callable, optional): function representing the theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): if it is set to ``True`` then, the ``k_norm`` is divided into bins and the mean and the standard deviation over each bin are derived and visualized on the plot. Note that the error bar represent 3 times the standard deviation. Defaults to False.

            file_name (str, optional): Defaults to "". Name used to save the figure. The available output formats depend on the backend being used.

        Returns:
            plot of the approximated structure factor.

        Example:

            .. literalinclude:: code/sf_baddour_example.py
                :lines: 30-35

            .. plot:: code/sf_baddour_example.py
                :include-source: False
        """
        return utils.plot_sf_hankel_quadrature(
            k_norm,
            sf,
            axis,
            k_norm_min,
            exact_sf,
            error_bar,
            file_name,
            **binning_params
        )
