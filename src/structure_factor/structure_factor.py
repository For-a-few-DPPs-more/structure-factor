r"""Class collecting estimators of the structure factor :math:`S(\mathbf{k})` of stationary point process given one realization encapsulated in a :py:class:`~structure_factor.point_pattern.PointPattern`.

**The available estimators:**

- :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`: The scattering intensity and the corresponding debiased versions.

- :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator`: The tapered or multitapered estimator and the corresponding debiased versions.

- :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator_isotropic`: Bartlett's isotropic estimator.

- :py:meth:`~structure_factor.structure_factor.StructureFactor.quadrature_estimator_isotropic`: Integral estimation using Hankel transform quadrature.

**The available plot methods:**

- :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_non_isotropic_estimator`: Visualize the results of :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator`, or :py:meth:`~structure_factor.structure_factor.StructureFactor.multitapered_estimator`.

- :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_isotropic_estimator`: Visualize the results of :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator_isotropic` or :py:meth:`~structure_factor.structure_factor.StructureFactor.quadrature_estimator_isotropic`.

For the theoretical derivation and definitions of these estimators, we refer to :cite:`HGBLR:22`.
"""

import warnings

import numpy as np

import structure_factor.plotting as plots
from structure_factor.point_pattern import PointPattern
from structure_factor.spatial_windows import BallWindow, BoxWindow
from structure_factor.tapered_estimators import (
    allowed_k_scattering_intensity,
    scattering_intensity,
    select_tapered_estimator,
)
from structure_factor.tapered_estimators_isotropic import (
    allowed_k_norm_bartlett_isotropic,
    bartlett_estimator,
)
from structure_factor.tapers import BartlettTaper
from structure_factor.transforms import RadiallySymmetricFourierTransform


class StructureFactor:
    r"""Implementation of various estimators of the structure factor :math:`S(\mathbf{k})` of a stationary point process given one realization encapsulated in the :py:attr:`~structure_factor.structure_factor.StructureFactor.point_pattern` attribute.

    .. todo::

        list attributes

    .. proof:definition::

        The structure factor :math:`S` of a d-dimensional stationary point process :math:`\mathcal{X}` with intensity :math:`\rho` is defined by,

        .. math::

            S(\mathbf{k}) = 1 + \rho \mathcal{F}(g-1)(\mathbf{k}),

        where :math:`\mathcal{F}` denotes the Fourier transform, :math:`g` the pair correlation function of :math:`\mathcal{X}`, :math:`\mathbf{k}` a wavevector of :math:`\mathbb{R}^d`.
        For more details we refer to :cite:`HGBLR:22`, (Section 2) or :cite:`Tor18`, (Section 2.1, equation (13)).
    """

    def __init__(self, point_pattern):
        r"""Initialize StructureFactor from ``point_pattern``.

        Args:
            point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Object of type point pattern which contains a realization ``point_pattern.points`` of a point process, the window where the points were simulated ``point_pattern.window`` and (optionally) the intensity of the point process ``point_pattern.intensity``.
        """
        assert isinstance(point_pattern, PointPattern)
        self.point_pattern = point_pattern

    @property
    def dimension(self):
        """Ambient dimension of the underlying point process."""
        return self.point_pattern.dimension

    def scattering_intensity(self, k=None, debiased=True, direct=True, **params):
        r"""Compute the scattering intensity :math:`\widehat{S}_{\mathrm{SI}}` estimate of the structure factor from one realization of a stationary point process encapsulated in the :py:attr:`~structure_factor.structure_factor.StructureFactor.point_pattern` attribute.

        Args:
            k (numpy.ndarray, optional): Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors where the scattering intensity is evaluated. If ``k=None`` and ``debiased=True``, the scattering intensity will be evaluated on the corresponding set of allowed wavevectors; In this case, the keyword arguments ``k_max``, and ``meshgrid_shape`` can be used. Defaults to None.

            debiased (bool, optional): Trigger the use of a debiased tapered estimator. Defaults to True. If ``debiased=True``, the estimator is debiased as follows,

                - if ``k=None``, the scattering intensity will be evaluated on the corresponding set of allowed wavevectors.
                - if ``k`` is not None and ``direct=True``, the direct debiased scattering intensity will be used,
                - if ``k`` is not None and ``direct=False``, the undirect debiased scattering intensity will be used.

            direct (bool, optional): If ``debiased`` is True, trigger the use of the direct/undirect debiased scattering intensity. Parameter related to ``debiased``. Defaults to True.

        Keyword Args:
            params (dict): Keyword arguments ``k_max`` and ``meshgrid_shape`` of :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity`, used when ``k=None`` and ``debiased=True``.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k: Wavevector(s) on which the scattering intensity has been evaluated.
                - estimation: Evaluation of the scattering intensity estimator of the structure factor at ``k``.

        Example:
             .. plot:: code/structure_factor/scattering_intensity.py
                :include-source: True

        .. proof:definition::

            The scattering intensity :math:`\widehat{S}_{\mathrm{SI}}` is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d` with intensity :math:`\rho`. It is computed from one realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W=\prod_{j=1}^d[-L_j/2, L_j/2]`.

            .. math::
                \widehat{S}_{\mathrm{SI}}(\mathbf{k}) =
                 \frac{1}{N}\left\lvert
                     \sum_{j=1}^N
                         \exp(- i \left\langle \mathbf{k}, \mathbf{x_j} \right\rangle)
                 \right\rvert^2 .

            For more details we refer to :cite:`HGBLR:22`, (Section 3.1).

        .. note::

            **Typical usage**

            - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::

            - :py:func:`~structure_factor.tapered_estimators.scattering_intensity`
            - :py:func:`~structure_factor.tapered_estimators.allowed_k_scattering_intensity`
            - :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator`
            - :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_non_isotropic_estimator`
            - :py:class:`~structure_factor.spatial_windows.BoxWindow`
            - :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`
        """
        if not isinstance(self.point_pattern.window, BoxWindow):
            warnings.warn(
                message="The observation window should be a BoxWindow to minimize the approximation error. Hint: use point_pattern.restrict_to_window."
            )

        window = self.point_pattern.window
        d = window.dimension
        if k is None:
            if not debiased:
                raise ValueError("debiased argument must be True when k is None .")
            L = np.diff(window.bounds)
            k = allowed_k_scattering_intensity(d, L, **params)

        estimated_sf_k = scattering_intensity(
            k, self.point_pattern, debiased=debiased, direct=direct
        )
        return k, estimated_sf_k

    # todo consider unpacking *tapers to cover single/multiple taper/s at once?
    def tapered_estimator(self, k, tapers, debiased=True, direct=True):
        r"""Compute the (multi)tapered estimator parametrized by ``tapers`` of the structure factor of a stationary point process given a realization encapsulated in ``point_pattern``.

        Args:
            k (numpy.ndarray): Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors where the tapered estimator is evaluated.

            t1, t2, ... : sequence of concrete tapers with methods ``.taper(x, window)`` corresponding to the taper function :math:`t(x, W)` , and ``.ft_taper(k, window)`` corresponding to the Fourier transform :math:`\mathcal{F}[t(\cdot, W)](k)` of the taper. See also :ref:`tapers`.

            debiased (bool, optional): Trigger the use of a debiased estimator. Defaults to True.

            direct (bool, optional): If ``debiased`` is True, trigger the use of the direct/undirect debiased tapered estimator. Parameter related to ``debiased``. Defaults to True.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k: Wavevector(s) on which the tapered estimator has been evaluated.
                - estimation: Evaluation of the tapered estimator at ``k``..

        .. note::

            Calling this method with its default arguments is equivalent to calling :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`.

        Example:
            .. plot:: code/structure_factor/multitapered_estimator.py
                :include-source: True

        .. proof:definition::

            The tapered estimator :math:`\widehat{S}_{\mathrm{TP}}(t, k)`, is an estimator of the structure factor :math:`S` of a stationary point process :math:`\mathcal{X} \subset \mathbb{R}^d` with intensity :math:`\rho`. It is computed from one realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a box window :math:`W`.

            .. math::

                \widehat{S}_{\mathrm{TP}}(t, \mathbf{k}) = \frac{1}{\rho} \left\lvert \sum_{j=1}^N t(x_j, W) \exp(- i \left\langle k, x_j \right\rangle)\right\rvert^2,

            If several tapers are used, a simple average of the corresponding tapered estimators is computed. The resulting estimator is call multitapered estimator.

            .. math::

                \widehat{S}_{\mathrm{MTP}}((t_{q})_{q=1}^P, \mathbf{k}) = \frac{1}{P}\sum_{q=1}^{P} \widehat{S}(t_{q}, \mathbf{k})

            where, :math:`(t_{q})_{q}` is a family of tapers supported on the observation window (satisfying some conditions), :math:`P` is the number of tapers used, and :math:`k \in \mathbb{R}^d`.
            For more details, we refer to :cite:`HGBLR:22`, (Section 3.1).

        .. note::

            **Typical usage**

            - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BoxWindow`, use the method :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BoxWindow`.

        .. seealso::

            - :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`
            - :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_non_isotropic_estimator`
            - :py:class:`~structure_factor.spatial_windows.BoxWindow`
            - :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`
            - :ref:`tapers`
            - :py:class:`~structure_factor.tapers.SineTaper`
            - :py:func:`~structure_factor.tapered_estimators.multitapered_estimator`
        """
        point_pattern = self.point_pattern
        n, d = k.shape
        if d != point_pattern.dimension:
            raise ValueError(
                f"k must be of size (n, d) where d=point_pattern.dimension. Given k {k.shape} and d d = {point_pattern.dimension}"
            )

        _tapers = list(tapers)
        estimator = select_tapered_estimator(debiased, direct)
        estimated_sf_k = np.zeros(n, dtype=float)
        for t in _tapers:
            estimated_sf_k += estimator(k, point_pattern, t)
        estimated_sf_k /= len(_tapers)
        return k, estimated_sf_k

    def bartlett_isotropic_estimator(self, k_norm=None, **params):
        r"""Compute Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` from one realization of an isotropic point process encapsulated in the :py:attr:`~structure_factor.structure_factor.StructureFactor.point_pattern` attribute.

        Args:
            k_norm (numpy.ndarray, optional): Array of wavenumbers of size :math:`n` where the estimator is to be evaluated. Defaults to None.

        Keyword Args:
            params (dict): Keyword argument ``nb_values`` of :py:func:`~structure_factor.tapered_estimators_isotropic.allowed_k_norm_bartlett_isotropic used when ``k_norm=None`` to specify the number of allowed wavenumbers to be considered.

        Returns:
            tuple(numpy.ndarray, numpy.ndarray):
                - k_norm: Wavenumber(s) on which Bartlett's isotropic estimator has been evaluated.
                - estimation: Evaluation(s) of Bartlett's isotropic estimator at ``k``.

        Example:
            .. plot:: code/structure_factor/bartlett_isotropic_estimator.py
                :include-source: True

        .. proof:definition::

            Bartlett's isotropic estimator :math:`\widehat{S}_{\mathrm{BI}}` is an estimator of the structure factor :math:`S` of a stationary isotropic point process :math:`\mathcal{X} \subset \mathbb{R}^d` with intensity :math:`\rho`. It is computed from one realization :math:`\mathcal{X}\cap W =\{\mathbf{x}_i\}_{i=1}^N` of :math:`\mathcal{X}` within a ball window :math:`W=B(\mathbf{0}, R)`.

            .. math::
                \widehat{S}_{\mathrm{BI}}(k)
                = 1 + \frac{ (2\pi)^{d/2} }{\rho |W| \omega_{d-1}} \sum_{ \substack{j, q =1 \\ j\neq q } }^{N }
                 \frac{1}{(k \|\mathbf{x}_j - \mathbf{x}_q\|_2)^{d/2 - 1}}
                J_{d/2 - 1}(k \|\mathbf{x}_j - \mathbf{x}_q\|_2).

            For more details, we refer to :cite:`HGBLR:22`, (Section 3.2).

        .. note::

            **Typical usage**

            - If the observation window is not a :py:class:`~structure_factor.spatial_windows.BallWindow`, use the method :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window` to extract a sub-sample in a :py:class:`~structure_factor.spatial_windows.BallWindow`.

        .. seealso::

            - :py:class:`~structure_factor.spatial_windows.BallWindow`
            - :py:meth:`~structure_factor.point_pattern.PointPattern.restrict_to_window`
            - :py:func:`~structure_factor.tapered_estimators_isotropic`
        """
        warnings.warn(
            message="The computation may take some time for a big number of points in the PointPattern. The complexity is quadratic in the number of points. Start by restricting the PointPattern to a smaller window using  PointPattern.restrict_to_window, then increasing the window progressively."
        )

        if not isinstance(self.point_pattern.window, BallWindow):
            warnings.warn(
                message="The observation window should be a BallWindow to minimize the approximation error. Hint: use point_pattern.restrict_to_window."
            )

        if k_norm is None:
            if not isinstance(self.point_pattern.window, BallWindow):
                raise TypeError(
                    "The observation window must be an instance of BallWindow. Hint: use point_pattern.restrict_to_window."
                )

            window = self.point_pattern.window
            d, r = window.dimension, window.radius
            k_norm = allowed_k_norm_bartlett_isotropic(
                dimension=d, radius=r, **params
            ).astype(float)

        sf_k_norm = bartlett_estimator(k_norm, self.point_pattern)
        return k_norm, sf_k_norm

    def quadrature_estimator_isotropic(
        self, pcf, k_norm=None, method="BaddourChouinard", **params
    ):
        # ? mettre k_nom avant pcf et donner le choix à l'utilisateur d'enter un None
        r"""Approximate the structure factor of a stationary isotropic point process at values ``k_norm``, given its pair correlation function ``pcf``, using a quadrature ``method``.

        .. warning::

            It only applies to point processes in even dimension :math:`d`, due to evaluations of the zeros of Bessel functions of order :math:`d / 2 - 1` that must integer-valued.

        Args:
            pcf (callable): Pair correlation function.

            k_norm (numpy.ndarray, optional): Vector of wavenumbers (i.e., norms of wavevectors) where the structure factor is to be evaluated. Optional if ``method="BaddourChouinard"`` (since this method evaluates the Hankel transform on a specific set of wavenumbers), but it is **non optional** if ``method="Ogata"``. Defaults to None.

            method (str, optional): Trigger the use of ``"BaddourChouinard"`` or ``"Ogata"`` quadrature to estimate the structure factor. Defaults to ``"BaddourChouinard"``,

                - if ``"BaddourChouinard"``: The Hankel transform is approximated using the Discrete Hankel transform :cite:`BaCh15`. See :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`

                - if ``"Ogata"``: The Hankel transform is approximated using Ogata quadrature :cite:`Oga05`. See :py:class:`~structure_factor.transforms.HankelTransformOgata`

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
            tuple (numpy.ndarray, numpy.ndarray):
                - k_norm: Vector of wavenumbers.
                - estimation: Evaluations of the structure factor on ``k_norm``.

        Example:
            .. plot:: code/structure_factor/hankel_quadrature.py
                :include-source: True

        .. proof:definition::

            The structure factor :math:`S` of a **stationary isotropic** point process :math:`\mathcal{X} \subset \mathbb{R}^d` with intensity :math:`\rho`, can be defined via the Hankel transform :math:`\mathcal{H}_{d/2 -1}` of order :math:`d/2 -1` as follows,

            .. math::

                S(\|\mathbf{k}\|_2)
                = 1 + \rho \frac{(2 \pi)^{d/2}}{\|\mathbf{k}\|_2^{d/2 -1}} \mathcal{H}_{d/2 -1}(\tilde g -1)(\|\mathbf{k}\|_2),
                \quad \tilde g: x \mapsto g(x) x^{d/2 -1},

            where, :math:`g` is the pair correlation function of :math:`\mathcal{X}`.

            This is a result of the relation between the Symmetric Fourier transform and the Hankel Transform.
            For more details, we refer to :cite:`HGBLR:22`, (Section 3.2).

        .. note::

            **Typical usage**

                1. Estimate the pair correlation function using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.estimate`.

                2. Clean and interpolate/extrapolate the resulting estimation using :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.interpolate` to get a **function**.

                3. Use the result as the input ``pcf``.

        .. seealso::

            - :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.estimate`
            - :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFonction.interpolate`
            - :py:meth:`~structure_factor.structure_factor.StructureFactor.plot_isotropic_estimator`
            - :py:class:`~structure_factor.spatial_windows`
            - :py:meth:`~structure_factor.point_pattern.PointPattern`
            - :py:class:`~structure_factor.transforms.HankelTransformBaddourChouinard`
            - :py:class:`~structure_factor.transforms.HankelTransformOgata`
        """
        assert callable(pcf)

        if method == "Ogata" and k_norm.all() is None:
            raise ValueError(
                "k_norm argument must be passed when using method='Ogata'."
            )

        r_max = params.setdefault("r_max", None)
        if method == "BaddourChouinard" and r_max is None:
            raise ValueError(
                "r_max keyword argument must be passed when using method='BaddourChouinard'."
            )

        ft = RadiallySymmetricFourierTransform(dimension=self.dimension)
        total_pcf = lambda r: pcf(r) - 1.0
        k_norm, ft_k = ft.transform(total_pcf, k_norm, method=method, **params)
        rho = self.point_pattern.intensity
        sf = 1.0 + rho * ft_k
        return k_norm, sf

    def plot_isotropic_estimator(
        self,
        k_norm,
        estimation,
        axis=None,
        scale="log",
        k_norm_min=None,
        exact_sf=None,
        color="grey",
        error_bar=False,
        label=r"$\widehat{S}$",
        file_name="",
        **binning_params,
    ):
        r"""Display the outputs of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.quadrature_estimator_isotropic`, or :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator_isotropic`.

        Args:
            k_norm (numpy.ndarray): Vector of wavenumbers (i.e., norms of wavevectors) on which the structure factor has been approximated.

            estimation (numpy.ndarray): Approximation(s) of the structure factor corresponding to ``k_norm``.

            axis (plt.Axes, optional): Support axis of the plot. Defaults to None.

            scale (str, optional): Trigger between plot scales of `see matplolib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Defaults to 'log'.

            k_norm_min (float, optional): Estimated lower bound of the wavenumbers. Defaults to None.

            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation``, are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``si`` are derived and visualized on the plot.  Note that each error bar corresponds to the mean :math:`\pm 3 \times` standard deviation. To specify the number of bins, add it as a keyword argument. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

        Keyword Args:
            binning_params: (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils_bin_statistics` as keyword arguments (except ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.

        Returns:
            plt.Axes: Plot of the approximated structure factor.
        """
        # todo normalize names of method and plotting routine
        return plots.plot_sf_hankel_quadrature(
            k_norm=k_norm,
            estimation=estimation,
            axis=axis,
            scale=scale,
            k_norm_min=k_norm_min,
            color=color,
            exact_sf=exact_sf,
            error_bar=error_bar,
            label=label,
            file_name=file_name,
            **binning_params,
        )

    def plot_non_isotropic_estimator(
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
        **binning_params,
    ):
        r"""Display the outputs of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.scattering_intensity`, :py:meth:`~structure_factor.structure_factor.StructureFactor.tapered_estimator`.

        Args:
            k (numpy.ndarray): Wavevector(s) on which the scattering intensity has been approximated. Array of size :math:`n \times d`  where :math:`d` is the dimension of the space, and :math:`n` is the number of wavevectors.

            estimation (numpy.ndarray): Approximated structure factor associated to `k`.

            axes (plt.Axes, optional): Support axes of the plots. Defaults to None.

            scale (str, optional): Trigger between plot scales of `see matplolib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xscale.html>`_. Defaults to "log".

            plot_type (str, optional): Type of the plot to visualize, "radial", "imshow", or "all". Defaults to "radial".

                - If "radial", the output is a 1D plot of `estimation` w.r.t. the norm(s) of `k`.
                - If "imshow" (option available only for a 2D point process), the output is a 2D color level plot.
                - If "all" (option available only for a 2D point process), the result contains 3 subplots: the point pattern (or a restriction to a specific window if ``window_res`` is set), the radial plot, and the color level plot. Note that the options "imshow" and "all" couldn't be used, if ``k`` couldn't be reshaped as a meshgrid.

            positive (bool, optional): If True, consider only the positive values of `estimation`. Defaults to False.

            exact_sf (callable, optional): Theoretical structure factor of the point process. Defaults to None.

            error_bar (bool, optional): If ``True``, ``k_norm`` and correspondingly ``estimation``, are divided into sub-intervals (bins). Over each bin, the mean and the standard deviation of ``estimation`` are derived and visualized on the plot. Note that each error bar corresponds to the mean :math:`\pm 3 \times` standard deviation. To specify the number of bins, add it as a keyword argument. For more details see :py:meth:`~structure_factor.utils._bin_statistics`. Defaults to False.

            rasterized (bool, optional): Rasterized option of `matlplotlib.plot <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html#:~:text=float-,rasterized,-bool>`_. Defaults to True.

            file_name (str, optional): Name used to save the figure. The available output formats depend on the backend being used. Defaults to "".

            window_res (:py:class:`~structure_factor.spatial_windows.AbstractSpatialWindow`, optional): New restriction window. Useful when the sample of points is large and "plt_type='all'", so for time and visualization purposes, it is better to restrict the plot of the point process to a smaller window. Defaults to None.

        Keyword Args:
            binning_params (dict): Used when ``error_bar=True``, by the method :py:meth:`~structure_factor.utils._bin_statistics` as keyword arguments (except ``"statistic"``) of `scipy.stats.binned_statistic <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html>`_.

        Returns:
            plt.Axes: Plot of the approximated structure factor.
        """
        k_norm = np.linalg.norm(k, axis=1)

        # unplot possible negative values
        if positive:
            si_ = estimation
            estimation = estimation[si_ >= 0]
            k_norm = k_norm[si_ >= 0]

        if plot_type == "radial":
            return plots.plot_estimation_showcase(
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
            return plots.plot_estimation_imshow(k_norm, estimation, axes, file_name)

        elif plot_type == "all":

            n_grid = int(np.sqrt(k_norm.shape[0]))
            grid_shape = (n_grid, n_grid)
            if self.dimension != 2:
                raise ValueError(
                    "The option 'all' is available only for 2D point processes."
                )

            estimation = estimation.reshape(grid_shape)
            k_norm = k_norm.reshape(grid_shape)
            return plots.plot_estimation_all(
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
