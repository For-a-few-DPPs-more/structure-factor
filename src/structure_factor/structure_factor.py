r"""Collection of estimators of the structure factor :math:`S(\mathbf{k})` of stationary point process given one realization encapsulated in a :py:class:`~structure_factor.point_pattern.PointPattern` together with the simulation window (:ref:`spatial_windows`), and the corresponding intensity.

**The available estimators:**

    - :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`: The scattering intensity and the corresponding debiased versions.
    - :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_periodogram`: The scaled tapered periodogram and the corresponding debiased versions.
    - :py:meth:`~structure_factor.structure_factor.StructureFactor.multitapered_periodogram`: The scaled multitapered periodogram and the corresponding debiased versions.
    - :py:meth:`~structure_factor.structure_factor.StructureFactor.bartlett_isotropic_estimator`: Bartlett's isotropic estimator.
    - :py:meth:`~structure_factor.structure_factor.StructureFactor.hankel_quadrature`: Integral estimation using Hankel transform quadrature.

For the theoretical derivation and definitions of these estimators, we refer to :cite:`DGRR:22`.
"""

import warnings
from isort import file

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

    #! doc done
    def scattering_intensity(self, k=None, debiased=True, direct=True, **params):
        r"""Compute the scattering intensity :math:`\widehat{S}_{\mathrm{SI}}` (or a debiased version) of the point process encapsulated in the ``PointPattern``.

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
                - estimation: Evaluation(s) of the scattering intensity or a debiased version at ``k``.

        Example:

             .. plot:: code/structure_factor/scattering_intensity.py
                :include-source: True

        .. proof:definition::

            The scattering intensity :math:`\widehat{S}_{\mathrm{SI}}` is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`. It is accessible from a realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W=\prod_{j=1}^d[-L_j/2, L_j/2]`.

            .. math::
                \widehat{S}_{\mathrm{SI}}(\mathbf{k}) =
                 \frac{1}{N}\left\lvert
                     \sum_{j=1}^N
                         \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                 \right\rvert^2 .

            For more details we refer to :cite:`DGRR:22`, (Section 3.1).

        .. note::

            **Typical usage**:
                - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::
            :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_spectral_estimator`,
            :py:class:`~structure_factor.spatial_windows.BoxWindow`,
            :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :py:func:`~structure_factor.utils.allowed_wave_vectors`.

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

    #! doc done
    def tapered_periodogram(self, k, taper=BartlettTaper, debiased=True, direct=True):
        r"""Compute the scaled tapered periodogram :math:`\widehat{S}_{\mathrm{TP}}` (or a debiased version :math:`\widehat{S}_{\mathrm{DDTP}}`, :math:`\widehat{S}_{\mathrm{UDTP}}`) of the point process encapsulated in the ``PointPattern`` for a specific choice of ``taper``.

        Args:
            k (np.ndarray): Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors where the scaled tapered periodogram is evaluated.

            taper (object, optional): Class with static method or instance with method ``.taper(x, window)`` corresponding to :math:`t(x, W)` (see :ref:`tapers`). Default to :py:class:`~structure_factor.tapers.BartlettTaper`

            debiased (bool, optional): Trigger the use of a debiased estimator. Default to True.

            direct (bool, optional): If ``debiased`` is True, trigger the use of the direct/undirect debiased scaled tapered periodogram. Parameter related to ``debiased``. Default to True.
        Returns:
            numpy.ndarray: Evaluation(s) of the scaled tapered periodogram or a debiased version at ``k``.

        Example:

             .. plot:: code/structure_factor/tapered_periodogram.py
                :include-source: True

        .. proof:definition::

            The scaled tapered periodogram :math:`\widehat{S}_{\mathrm{TP}}(t, k)`, is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`. It is accessible from a realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W`.

            .. math::

                \widehat{S}_{\mathrm{TP}}(t, \mathbf{k}) = \frac{1}{\rho} \left\lvert \sum_{j=1}^N t(x_j, W) \exp(- i \left\langle k, x_j \right\rangle)\right\rvert^2,

            where, :math:`t` is a taper supported on the observation window (satisfying some conditions) and :math:`k \in \mathbb{R}^d`.
            For more details we refer to :cite:`DGRR:22`, (Section 3.1).

        .. note::

            **Typical usage**:
                - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::
            :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_spectral_estimator`,
            :py:class:`~structure_factor.spatial_windows.BoxWindow`,
            :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :ref:`tapers`, :py:class:`~structure_factor.tapers.BartlettTaper`, :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_core`, :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_debiased_direct`, :py:func:`~structure_factor.spectral_estimators.tapered_spectral_estimator_debiased_undirect`.


        """
        estimator = select_tapered_spectral_estimator(debiased, direct)
        estimation = estimator(k, self.point_pattern, taper)
        return estimation

    #! doc done
    def multitapered_periodogram(
        self, k, tapers=None, debiased=True, direct=True, **params
    ):
        r"""Compute the scaled multitapered periodogram :math:`\widehat{S}_{\mathrm{MTP}}` (or a debiased version :math:`\widehat{S}_{\mathrm{MDDTP}}`, :math:`\widehat{S}_{\mathrm{MUDTP}}`) of the point process encapsulated in the ``PointPattern`` for the family of tapers  ``tapers``.

        Args:
            k (np.ndarray): Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors where the scaled tapered periodogram is evaluated.

            tapers (list, optional): List of tapers. Defaults to :py:class:`~structure_factor.tapers.SineTaper`. Each taper is an instance with two methods:

                - ``.taper(x, window)`` corresponding to the taper function :math:`t(x, W)`.

                - ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the taper function, used if ``debiased`` is True.

            debiased (bool, optional): Trigger the use of a debiased estimator. Default to True.

            direct (bool, optional): If ``debiased`` is True, trigger the use of the direct/undirect debiased scaled tapered periodogram. Parameter related to ``debiased``. Default to True.
        Keyword Args:
            params (dict): Keyword argument ``p_component_max`` of :py:func:`~structure_factor.utils.taper_grid_generator`. Maximum component of the parameters :math:`p` of the family of :py:class:`~structure_factor.tapers.SineTaper`. Intuitively the number of taper used is :math:`P=\mathrm{p\_component\_max}^d`. Used only when ``tapers=None``. See :py:func:`~structure_factor.utils.tapered_generator`. Default to 2.
        Returns:
            numpy.ndarray: Evaluation(s) of the scaled multitapered periodogram or a debiased version at ``k``.

        Example:

            .. plot:: code/structure_factor/multitapered_periodogram.py
                :include-source: True

        .. proof:definition::

            The scaled multitapered periodogram :math:`\widehat{S}_{\mathrm{MTP}}(t, k)`, is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`. It is accessible from a realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W`.

            .. math::

                \widehat{S}_{ \mathrm{MTP}}((t_{q})_{q=1}^P, \mathbf{k}) = \frac{1}{P}\sum_{q=1}^{P} \widehat{S}(t_{q}, \mathbf{k})


            where, :math:`(t_{q})_{q}` is a family of tapers supported on the observation window (satisfying some conditions), :math:`P` is the number of tapers used, and :math:`k \in \mathbb{R}^d`.
            For more details, we refer to :cite:`DGRR:22`, (Section 3.1).

        .. note::

            **Typical usage**:
                - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::
            :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_spectral_estimator`,
            :py:class:`~structure_factor.spatial_windows.BoxWindow`,
            :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :ref:`tapers`, :py:class:`~structure_factor.tapers.SineTaper`, :py:func:`~structure_factor.spectral_estimators.multitapered_spectral_estimator`.

        """
        d = self.point_pattern.dimension
        if tapers is None:
            tapers = utils.taper_grid_generator(d=d, taper_p=SineTaper, **params)
        estimation = multitapered_spectral_estimator(
            k,
            self.point_pattern,
            *tapers,
            debiased=debiased,
            direct=direct,
        )
        return estimation

    #! doc done maybe add example
    def plot_spectral_estimator(
        self,
        k,
        estimation,
        axes=None,
        scale="log",
        plot_type="radial",
        positive=False,
        exact_sf=None,
        error_bar=False,
        label=r"$\widehat{S}$",
        rasterized=True,
        file_name="",
        window_res=None,
        **binning_params
    ):
        r"""Display the outputs of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_periodogram`, or :py:meth:`~structure_factor.structure_factor.StructureFactor.multitapered_periodogram`.

        Args:
            k (numpy.ndarray): Wavevector(s) on which the scattering intensity has been approximated. Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors.

            estimation (numpy.array): Approximated structure factor associated to `k`.

            axes (matplotlib.axis, optional): Support axes of the plots. Defaults to None.

            scale(str, optional): Trigger between plot scales of `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Default to "log".

            plot_type (str, optional): Type of the plot to visualize, "radial", "imshow", or "all". Defaults to "radial".
                    - If "radial", the output is a 1D plot of `estimation` w.r.t. the norm(s) of `k`.
                    - If "imshow" (option available only for a 2D point process), the output is a 2D color level plot.
                    - If "all" (option available only for a 2D point process), the result contains 3 subplots: the point pattern (or a restriction to a specific window if ``window_res`` is set), the radial plot, and the color level plot. Note that the options "imshow" and "all" couldn't be used, if ``k`` couldn't be reshaped as a meshgrid.

            positive (bool, optional): If True, plots only the positive values of `estimation`. Default to False.
            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation``, are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``estimation`` are derived and visualized on the plot. Note that each error bar corresponds to the mean :math:`\pm 3 \times` standard deviation. To specify the number of bins, add it as a keyword argument. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

            rasterized (bool, optional): Rasterized option of `matlplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#:~:text=float-,rasterized,-bool>`_. Default to True.

            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

            window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): New restriction window. Useful when the sample of points is large and "plt_type='all'", so for time and visualization purposes, it is better to restrict the plot of the point process to a smaller window. Defaults to None.

        Keyword Args:
            binning_params (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils._bin_statistics` as keyword arguments (except ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.

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
            return utils.plot_estimation_showcase(
                k_norm,
                estimation,
                axis=axes,
                scale=scale,
                exact_sf=exact_sf,
                error_bar=error_bar,
                label=label,
                rasterized=rasterized,
                file_name=file_name,
                **binning_params,
            )

        elif plot_type == "imshow":

            if self.dimension != 2:
                raise ValueError(
                    "This plot option is available only for a 2D point process. Hint: Use `plot_type ='radial'`."
                )
            n_grid = int(np.sqrt(k_norm.shape[0]))
            if k_norm.shape[0] != n_grid ** 2:
                raise ValueError(
                    "Wavevectors couldn't be reshaped as meshgrid. Hint: the square root of the number of wavevectors should be an integer."
                )
            grid_shape = (n_grid, n_grid)
            estimation = estimation.reshape(grid_shape)
            k_norm = k_norm.reshape(grid_shape)
            return utils.plot_estimation_imshow(k_norm, estimation, axes, file_name)

        elif plot_type == "all":

            n_grid = int(np.sqrt(k_norm.shape[0]))
            grid_shape = (n_grid, n_grid)
            if self.dimension != 2:
                raise ValueError(
                    "The option 'all' is available only for 2D point processes."
                )

            estimation = estimation.reshape(grid_shape)
            k_norm = k_norm.reshape(grid_shape)
            return utils.plot_estimation_all(
                self.point_pattern,
                k_norm,
                estimation,
                exact_sf=exact_sf,
                error_bar=error_bar,
                label=label,
                rasterized=rasterized,
                file_name=file_name,
                window_res=window_res,
                scale=scale,
                **binning_params,
            )
        else:
            raise ValueError(
                "plot_type must be chosen among ('all', 'radial', 'imshow')."
            )

    #! doc done
    def bartlett_isotropic_estimator(self, k_norm=None, **params):
        r"""Compute Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` of the point process (isotropic) encapsulated in the ``PointPattern``.

        Args:
            k_norm (np.ndarray, optional): n rows of wavenumbers where the estimator is to be evaluated. If ``k_norm=None`` (recommended)and the space's dimension is an even number, the estimator will be evaluated on the corresponding set of allowed wavenumbers; In this case, the parameters ``n_allowed_k_norm`` allows to specify the number of allowed wavenumbers. See :py:func:`~structure_factor.isotropic_estimator.allowed_k_norm`. Defaults to None.
        Keyword Args:
            params (dict): Keyword argument ``n_allowed_k_norm`` of :py:func:`~structure_factor.isotropic_estimator.bartlett_estimator`. Used when ``k_norm=None`` to specify the number of allowed wavenumbers to be used.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k_norm: Wavenumber(s) on which Bartlett's isotropic estimator has been evaluated.
                - estimation: Evaluation(s) of Bartlett's isotropic estimator at ``k``.

        Example:

            .. plot:: code/structure_factor/bartlett_isotropic_estimator.py
                :include-source: True

        .. proof:definition::

            Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` is an estimator of the structure factor :math:`S` of a stationary isotropic point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`. It is accessible from a realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a ball window :math:`W=B(\mathbf{0}, R)`.

            .. math::
                \widehat{S}_{\mathrm{BI}}(k) =1 + \frac{ (2\pi)^{d/2} }{\rho \mathcal{L}^d(W) \omega_{d-1}} \sum_{ \substack{j, q =1 \\ j\neq q } }^{N }
                 \frac{1}{(k \|\mathbf{x}_j - \mathbf{x}_q\|_2)^{d/2 - 1}}
                J_{d/2 - 1}(k \|\mathbf{x}_j - \mathbf{x}_q\|_2).

            For more details, we refer to :cite:`DGRR:22`, (Section 3.2).

        .. note::

            **Typical usage**:
                - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BallWindow`, use the method :py:class:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BallWindow`.

        .. seealso::
            :py:class:`~structure_factor.spatial_windows.BallWindow`,
            :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`, :py:func:`~structure_factor.isotropic_estimator`.

        """
        window = self.point_pattern.window
        warnings.warn(
            message="The computation may take some time for a big number of points in the PointPattern. The complexity is quadratic in the number of points. Start by restricting the PointPattern to a smaller window using  PointPattern.restrict_to_window, then increasing the window progressively."
        )
        if not isinstance(window, BallWindow):
            warnings.warn(
                message="The window should be a BallWindow to minimize the approximation error. Hint: use PointPattern.restrict_to_window."
            )

        k_norm, estimation = ise.bartlett_estimator(
            point_pattern=self.point_pattern, k_norm=k_norm, **params
        )
        return k_norm, estimation

    #! doc done maybe change def
    def hankel_quadrature(self, pcf, k_norm=None, method="BaddourChouinard", **params):
        r"""Approximate the structure factor of the point process encapsulated in ``point_pattern`` (only for stationary isotropic point processes), using specific approximations of the Hankel transform.

        .. warning::

            This method is actually applicable for 2-dimensional point processes.

        Args:
            pcf (callable): Pair correlation function.
            k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e., norms of wavevectors) where the structure factor is to be evaluated. Optional if ``method="BaddourChouinard"`` (since this method evaluates the Hankel transform on a specific set of wavenumbers), but it is **non optional** if ``method="Ogata"``. Defaults to None.
            method (str, optional): Trigger the use of ``"BaddourChouinard"`` or ``"Ogata"`` quadrature to estimate the structure factor. Defaults to ``"BaddourChouinard"``,

                - if ``"BaddourChouinard"``: The Hankel transform is approximated using the Discrete Hankel transform :cite:`BaCh15`. See :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`,
                - if ``"Ogata"``: The Hankel transform is approximated using Ogata quadrature :cite:`Oga05`. See :py:class:`~structure_factor.transforms.HankelTransformOgata`.

        Keyword Args:
            params (dict): Keyword arguments passed to the corresponding Hankel transformer selected according to the input argument ``method``.

                - ``method == "Ogata"``, see :py:meth:`~structure_factor.transforms.HankelTransformOgata.compute_transformation_parameters`
                    - ``step_size``
                    - ``nb_points``
                - ``method == "BaddourChouinard"``, see :py:meth:`~structure_factor.transforms.HankelTransformBaddourChouinard.compute_transformation_parameters`
                    - ``r_max``
                    - ``nb_points``
                    - ``interpolotation`` dictionnary containing the keyword arguments of `scipy.integrate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ parameters.

        Returns:
            tuple (np.array, np.array):
                - k_norm: Vector of wavenumbers.
                - estimation: Evaluations of the structure factor on ``k_norm``.

        Example:
            .. plot:: code/structure_factor/hankel_quadrature.py
                :include-source: True

        .. proof:definition::

            The structure factor :math:`S` of a **stationary isotropic** point process :math:`\mathcal{X} \subset \mathbb{R}^d` of intensity :math:`\rho`, can be defined via the Hankel transform :math:`\mathcal{H}_{d/2 -1}` of order :math:`d/2 -1` as follows,

            .. math::

                S(\|\mathbf{k}\|_2)
                = 1 + \rho \frac{(2 \pi)^{d/2}}{\|\mathbf{k}\|_2^{d/2 -1}} \mathcal{H}_{d/2 -1}(\tilde g -1)(\|\mathbf{k}\|_2),
                \quad \tilde g: x \mapsto g(x) x^{d/2 -1},

            where, :math:`g` is the pair correlation function of :math:`\mathcal{X}`.
            This is a result of the relation between the Symmetric Fourier transform and the Hankel Transform.
            For more details, we refer to :cite:`DGRR:22`, (Section 3.2).

        .. note::

            **Typical usage**:
                1. Estimate the pair correlation function using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.estimate`.

                2. Clean and interpolate/extrapolate the resulting estimation using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.interpolate` to get a **function**.

                3. Use the result as the input ``pcf``.

        .. seealso::
            :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.estimate`, :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.interpolate`, :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_isotropic_estimator`, :py:class:`~structure_factor.spatial_windows`,
            :py:meth:`~structure_factor.point_pattern.PointPattern`, :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`, :py:class:`~structure_factor.transforms.HankelTransformOgata`.

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
        estimation = 1.0 + self.point_pattern.intensity * ft_k
        return k_norm, estimation

    #! doc done maybe add example
    def plot_isotropic_estimator(
        self,
        k_norm,
        estimation,
        axis=None,
        scale="log",
        k_norm_min=None,
        exact_sf=None,
        error_bar=False,
        label=r"$\widehat{S}$",
        file_name="",
        **binning_params
    ):
        r"""Display the outputs of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.hankel_quadrature`, or :py:meth:`~structure_factor.structure_factor.StructureFactor.bartlett_isotropic_estimator`

        Args:
            k_norm (np.array): Vector of wavenumbers (i.e., norms of wavevectors) on which the structure factor has been approximated.

            estimation (np.array): Approximation(s) of the structure factor corresponding to ``k_norm``.

            axis (matplotlib.axis, optional): Support axis of the plot. Defaults to None.

            scale(str, optional): Trigger between plot scales of `matplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Default to 'log'.

            k_norm_min (float, optional): Estimated lower bound of the wavenumbers. Defaults to None.

            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation``, are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot.  Note that each error bar corresponds to the mean :math:`\pm 3 \times` standard deviation. To specify the number of bins, add it as a keyword argument. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        Keyword Args:
            binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.


        Returns:
            matplotlib.plot: Plot of the approximated structure factor.

        """
        return utils.plot_sf_hankel_quadrature(
            k_norm=k_norm,
            estimation=estimation,
            axis=axis,
            scale=scale,
            k_norm_min=k_norm_min,
            exact_sf=exact_sf,
            error_bar=error_bar,
            label=label,
            file_name=file_name,
            **binning_params,
        )
