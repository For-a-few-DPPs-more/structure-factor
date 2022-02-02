r"""Collection of estimators of the structure factor :math:`S(\mathbf{k})` of stationary point process given one realization encapsulated in a :py:class:`~structure_factor.point_pattern.PointPattern` together with the simulation window (:ref:`spatial_windows`), and the corresponding intensity.

**The available estimators are:**

    - The scattering intensity and the corresponding debiased versions: :func:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`.
    - The scaled tapered periodogram and the corresponding debiased versions: :func:`~structure_factor.structure_factor.StructureFactor.tapered_periodogram`.
    - The scaled multitapered periodogram and the corresponding debiased versions: :func:`~structure_factor.structure_factor.StructureFactor.multitapered_periodogram`.
    - Bartlett's isotropic estimator: :func:`~structure_factor.structure_factor.StructureFactor.bartlett_isotropic_estimator`.
    - Integral estimation using Hankel transform quadrature: :func:`~structure_factor.structure_factor.StructureFactor.hankel_quadrature`.

For the theoretical derivation of these estimators, we refer to :cite:`DGRR:22`.
"""
# todo the above spatial_window doesn't in the doc refers see why
import warnings

import numpy as np

import structure_factor.isotropic_estimator as ise
import structure_factor.utils as utils
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.spectral_estimators import (
    multitapered_spectral_estimator,
    select_tapered_spectral_estimator,
)
from structure_factor.tapers import BartlettTaper, SineTaper
from structure_factor.transforms import RadiallySymmetricFourierTransform


class StructureFactor:
    r"""Implementation of various estimators of the structure factor :math:`S(\mathbf{k})` of a :py:class:`~structure_factor.point_pattern.PointPattern`.

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type :py:class:`~structure_factor.point_pattern.PointPattern` containing a realization of a point process, the observation window, and (optionally) the intensity of the point process (see :py:class:`~structure_factor.point_pattern.PointPattern`).

    .. proof:definition::

        The structure factor :math:`S` of a d-dimensional stationary point process :math:`\mathcal{X}` with intensity :math:`\rho` is defined by,

        .. math::

            S(\mathbf{k}) = 1 + \rho \mathcal{F}(g-1)(\mathbf{k}),

        where :math:`\mathcal{F}` denotes the Fourier transform, :math:`g` the pair correlation function of :math:`\mathcal{X}`, :math:`\mathbf{k}` a wavevector of :math:`\mathbb{R}^d`.
        For more details we refer to :cite:`DGRR:22`, (Section 2) or :cite:`Tor18`, (Section 2.1, equation (13)).
    """

    def __init__(self, point_pattern):
        r"""Initialize StructureFactor from ``point_pattern``.
        Args:
            point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.
        """
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = point_pattern  # the point pattern
        self.k_norm_min = None  # minimal bounds on the wavenumbers for Ogata method

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    #! doc done untill example
    def scattering_intensity(self, k=None, debiased=True, direct=True, **params):
        r"""Compute the scattering intensity (an estimator of the structure factor) of the point process encapsulated in the ``PointPattern``.

        Args:
            k (np.ndarray, optional): Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors where the scattering intensity is evaluated. If ``k=None`` and ``debiased=True``, the scattering intensity will be evaluated on the corresponding set of allowed wavevectors; In this case, the parameters ``k_max``, and ``meshgrid_shape`` could be used. See :py:attr:`~structure_factor.utils.allowed_wave_vectors`, for more details about ``k_max``, and ``meshgrid_shape``. Defaults to None.
            debiased (bool, optional): Trigger the use of a debiased tapered estimator. Default to True. If ``debiased=True``, the estimator is debiased as follows,

                - if ``k=None``, the scattering intensity will be evaluated on the corresponding set of allowed wavevectors.
                - if ``k`` is not None and ``direct=True``, the direct debiased scattering intensity will be used,
                - if ``k`` is not None and ``direct=False``, the undirect debiased scattering intensity will be used.

            direct (bool, optional): If ``debiased`` is True, trigger the use of the direct/undirect debiased scattering intensity. Parameter related to ``debiased``. Default to True.
        Keyword Args:
            params (dict): Keyword arguments ``k_max`` and ``meshgrid_shape`` of :py:attr:`~structure_factor.utils.allowed_wave_vectors`. Used when ``k=None`` and ``debiased=True``.
        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k: Wavevector(s) on which the scattering intensity has been evaluated.
                - estimation: Evaluation(s) of the scattering intensity corresponding to ``k``.

        Example:

             .. plot:: code/structure_factor/scattering_intensity.py
                :include-source: True

        .. proof:definition::

            The scattering intensity :math:`\widehat{S}_{\mathrm{SI}}` is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d`. It is accessible from a realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W=\prod_{j=1}^d[-L_j/2, L_j/2]`.

            .. math::
                \widehat{S}_{\mathrm{SI}}(\mathbf{k}) =
                 \frac{1}{N}\left\lvert
                     \sum_{j=1}^N
                         \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                 \right\rvert^2 .

            For more details we refer to :cite:`DGRR:22`, (Section 3).

        .. note::

            **Typical usage**:
                - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::
            :py:class:`~structure_factor.spatial_windows.BoxWindow`,
            :py:class:`~structure_factor.point_pattern.PointPattern`, :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :py:func:`~structure_factor.utils.allowed_wave_vectors`.

        """

        point_pattern = self.point_pattern
        d = point_pattern.dimension

        window = point_pattern.window
        if not isinstance(window, BoxWindow):
            warnings.warn(
                message="The window should be a BoxWindow to minimize the approximation error. Hint: use PointPattern.restrict_to_window."
            )

        if k is None:
            if not debiased:
                raise ValueError("when k is None debiased must be True.")
            L = np.diff(window.bounds)
            k = utils.allowed_wave_vectors(d, L, **params)

        elif k.shape[1] != d:
            raise ValueError(
                "`k` should have d columns, where d is the dimension of the ambient space where the points forming the point pattern live."
            )

        estimation = self.tapered_periodogram(
            k, taper=BartlettTaper(), debiased=debiased, direct=direct
        )

        return k, estimation

    def tapered_periodogram(self, k, taper, debiased=True, direct=True):
        """todo

        [extended_summary]

        Args:
            k ([type]): [description]
            taper ([type]): [description]
            debiased (bool, optional): [description]. Defaults to True.
            direct (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """
        estimator = select_tapered_spectral_estimator(debiased, direct)
        sf = estimator(k, self.point_pattern, taper)
        return sf

    def multitapered_periodogram(
        self, k, tapers=None, debiased=True, direct=True, **params
    ):
        """[summary]

        [extended_summary]

        Args:
            k ([type]): [description]
            tapers ([type], optional): [description]. Defaults to None.
            debiased (bool, optional): [description]. Defaults to True.
            direct (bool, optional): [description]. Defaults to True.
        Keyword Args:
            params: P for grid sine taper
        Returns:
            [type]: [description]
        """
        d = self.point_pattern.dimension
        if tapers is None:
            tapers = utils.taper_grid_generator(d=d, taper_p=SineTaper, **params)
        sf = multitapered_spectral_estimator(
            k,
            self.point_pattern,
            *tapers,
            debiased=debiased,
            direct=direct,
        )
        return sf

    def plot_spectral_estimator(
        self,
        k,
        estimation,
        axes=None,
        plot_type="radial",
        positive=False,
        exact_sf=None,
        error_bar=False,
        label=r"$\widehat{S}$",
        file_name="",
        window_res=None,
        **binning_params
    ):
        """Visualize the results of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`.

        Args:
            k (numpy.array): Wavevector(s) on which the scattering intensity has been approximated.
            estimation (numpy.array): Approximated scattering intensity associated to `k`.
            axes (matplotlib.axis, optional): Support axes of the plots. Defaults to None.
            plot_type (str, optional): ("radial", "imshow", "all"). Type of the plot to visualize. Defaults to "radial".
                    - If "radial", the output is a loglog plot.
                    - If "imshow" (option available only for a 2D point process), the output is a (2D) color level plot.
                    - If "all" (option available only for a 2D point process), the result contains 3 subplots: the point pattern (or a restriction to a specific window if ``window_res`` is set), the loglog radial plot, and the color level plot. Note that the options "imshow" and "all" couldn't be used, if ``k`` is not a meshgrid.

            positive (bool): If True, plots only the positive values of si.
            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``estimation`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.
            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".
            window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): New restriction window. It is useful when the sample of points is large, so for time and visualization purposes, it is better to restrict the plot of the point process to a smaller window. Defaults to None.

        Keyword Args:
            binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.

        Returns:
            matplotlib.plot: Plot of the approximated structure factor.

        """
        k_norm = np.linalg.norm(k, axis=1)

        # unplot possible negative values
        if positive:
            si_ = estimation
            estimation = estimation[si_ >= 0]
            k_norm = k_norm[si_ >= 0]

        if plot_type == "radial":
            return utils.plot_si_showcase(
                k_norm,
                estimation,
                axes,
                exact_sf,
                error_bar,
                label,
                file_name,
                **binning_params,
            )
        #! todo k may be already a meshgrid and add a warning else
        elif plot_type == "imshow":
            n_grid = int(np.sqrt(k_norm.shape[0]))
            grid_shape = (n_grid, n_grid)
            if self.dimension != 2:
                raise ValueError(
                    "This plot option is adapted only for a 2D point process. Please use plot_type ='radial'."
                )

            estimation = estimation.reshape(grid_shape)
            k_norm = k_norm.reshape(grid_shape)
            return utils.plot_si_imshow(k_norm, estimation, axes, file_name)

        elif plot_type == "all":

            n_grid = int(np.sqrt(k_norm.shape[0]))
            grid_shape = (n_grid, n_grid)
            if self.dimension != 2:
                raise ValueError(
                    "The option 'all' is available for 2D point processes."
                )

            estimation = estimation.reshape(grid_shape)
            k_norm = k_norm.reshape(grid_shape)
            return utils.plot_si_all(
                self.point_pattern,
                k_norm,
                estimation,
                exact_sf,
                error_bar,
                label,
                file_name,
                window_res,
                **binning_params,
            )
        else:
            raise ValueError(
                "plot_type must be chosen among ('all', 'radial', 'imshow')."
            )

    def bartlett_isotropic_estimator(self, k_norm=None, **params):
        """todo

        [extended_summary]

        Args:
            k_norm ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        window = self.point_pattern.window
        warnings.warn(
            message="The computation may take some time for a big number of points in the PointPattern. The complexity is quadratic in the number of points. Start by restricting the PointPattern to a smaller window using  PointPattern.restrict_to_window, then increasing the window progressively."
        )
        if not isinstance(window, BallWindow):
            warnings.warn(
                message="The window should be a BallWindow to minimize the approximation error. Hint: use PointPattern.restrict_to_window."
            )

        k_norm, sf = ise.bartlett_estimator(
            point_pattern=self.point_pattern, k_norm=k_norm, **params
        )
        return k_norm, sf

    def hankel_quadrature(self, pcf, k_norm=None, method="BaddourChouinard", **params):
        # ? mettre k_nom avant pcf et donner le choix à l'utilisateur d'enter un None
        r"""Approximate the structure factor of the point process encapsulated in ``point_pattern`` (only for stationary isotropic point processes), using specific approximations of the Hankel transform.
        .. warning::
            This method is actually applicable for 2-dimensional point processes.

        Args:
            pcf (callable): Radially symmetric pair correlation function.
            k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e., norms of wave vectors) where the structure factor is to be evaluated. Optional if ``method="BaddourChouinard"`` (since this method evaluates the Hankel transform on a specific vector, see :cite:`BaCh15`), but it is **non optional** if ``method="Ogata"``. Defaults to None.
            method (str, optional): Choose between ``"BaddourChouinard"`` or ``"Ogata"``. Defaults to ``"BaddourChouinard"``. Selects the method to be used to compute the Hankel transform corresponding to the symmetric Fourier transform of ``pcf -1``,

                - if ``"BaddourChouinard"``: The Hankel transform is approximated using the Discrete Hankel transform :cite:`BaCh15`. See :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`,
                - if ``"Ogata"``: The Hankel transform is approximated using Ogata quadrature :cite:`Oga05`. See :py:class:`~structure_factor.transforms.HankelTransformOgata`.

        Keyword Args:
            params (dict): Keyword arguments passed to the corresponding Hankel transformer selected according to the ``method`` argument.

                - ``method == "Ogata"``, see :py:meth:`~structure_factor.transforms.HankelTransformOgata.compute_transformation_parameters`
                    - ``step_size``
                    - ``nb_points``
                - ``method == "BaddourChouinard"``, see :py:meth:`~structure_factor.transforms.HankelTransformBaddourChouinard.compute_transformation_parameters`
                    - ``r_max``
                    - ``nb_points``
                    - ``interpolotation`` dictionnary containing the keyword arguments of `scipy.integrate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ parameters.

        Returns:
            tuple (np.ndarray, np.ndarray):
                - k_norm: Vector of wavenumbers.
                - sf: Evaluations of the structure factor on ``k_norm``.

        Example:
            .. literalinclude:: code/sf_baddour_example.py
                :lines: 1-28
                :emphasize-lines: 23-28

        .. proof:definition::

            The structure factor :math:`S` of a **stationary isotropic** point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`, can be defined via the Hankel transform :math:`\mathcal{H}_{d/2 -1}` of order :math:`d/2 -1` as follows,

            .. math::

                S(\|\mathbf{k}\|)
                = 1 + \rho \frac{(2 \pi)^{d/2}}{\|\mathbf{k}\|^{d/2 -1}} \mathcal{H}_{d/2 -1}(\tilde g -1)(\|\mathbf{k}\|),
                \quad \tilde g: x \mapsto g(x) x^{d/2 -1},

            where, :math:`g` is the pair correlation function of :math:`\mathcal{X}`.
            This is a result of the relation between the Symmetric Fourier transform and the Hankel Transform.

        .. note::
            **Typical usage**:
                1. Estimate the pair correlation function using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.estimate`.

                2. Clean and interpolate the resulting estimation using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.interpolate` to get a **function**.

                3. Pass the resulting interpolated function to :py:meth:`hankel_quadrature` to get an approximation of the structure factor of the point process.
        """
        if self.dimension != 2:
            warnings.warn(
                message="This method is actually applicable for 2-dimensional point processes",
                category=DeprecationWarning,
            )
        assert callable(pcf)

        if method == "Ogata" and k_norm.all() is None:
            raise ValueError(
                "k_norm is not optional while using method='Ogata'. Please provide a vector k_norm in the input. "
            )
        params.setdefault("r_max", None)
        if method == "BaddourChouinard" and params["r_max"] is None:
            raise ValueError(
                "r_max is not optional while using method='BaddourChouinard'. Please specify r_max in the input. "
            )
        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        k_norm, ft_k = ft.transform(total_pcf, k_norm, method=method, **params)
        if method == "Ogata" and params["r_max"] is not None:
            params.setdefault("step_size", 0.1)
            step_size = params["step_size"]
            # todo does it really metter k_norm_min?
            self.k_norm_min = utils._compute_k_min(
                r_max=params["r_max"], step_size=step_size
            )
        sf = 1.0 + self.point_pattern.intensity * ft_k
        return k_norm, sf

    def plot_isotropic_estimator(
        self,
        k_norm,
        sf,
        axis=None,
        k_norm_min=None,
        exact_sf=None,
        error_bar=False,
        label=r"$\widehat{S}$",
        file_name="",
        **binning_params
    ):
        r"""Display the output of :py:meth:`hankel_quadrature`.

        Args:
            k_norm (np.array): Vector of wavenumbers (i.e., norms of waves) on which the structure factor has been approximated.
            sf (np.array): Approximation of the structure factor corresponding to ``k_norm``.
            axis (matplotlib.axis, optional): Support axis of the plots. Defaults to None.
            k_norm_min (float, optional): Estimated lower bound of the wavenumbers (only when ``sf`` was approximated using **Ogata quadrature**). Defaults to None.
            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.
            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``si`` are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot. Note that each error bar corresponds to the mean +/- 3 standard deviation. To specify the number of bins, add it to the kwargs argument ``binning_params``. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.
            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        Keyword Args:
            binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of ``scipy.stats.binned_statistic``.

        Returns:
            matplotlib.plot: Plot the output of :py:meth:`~structure_factor.structure_factor.StructureFactor.hankel_quadrature`.

        """
        return utils.plot_sf_hankel_quadrature(
            k_norm=k_norm,
            sf=sf,
            axis=axis,
            k_norm_min=k_norm_min,
            exact_sf=exact_sf,
            error_bar=error_bar,
            label=label,
            file_name=file_name,
            **binning_params,
        )
