import warnings

import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import scipy.interpolate as interpolate
from spatstat_interface.interface import SpatstatInterface

import structure_factor.utils as utils

# ? do we need to create a class with only statimethods, these could be simple functions


class PairCorrelationFunction:
    @staticmethod
    def estimate(point_pattern, method="fv", install_spatstat=False, **params):
        r"""Estimate the pair correlation function (pcf) of the point process encapsulated in ``point_pattern`` (only for stationary isotropic point processes of :math:`\mathbb{R}^2`). The available methods are the methods ``spastat.core.pcf_ppp`` and ``spastat.core.pcf_fv`` of the `R` package `spatstat <https://github.com/spatstat/spatstat>`_.

        .. warning::

            This function requires the `R programming language <https://cran.r-project.org/>`_ to be installed on your local machine since it relies on the Python package `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_. This doesn't require any knowledge of the programming language R.

        Args:
            method (str, optional): Choose between ``"ppp"`` or ``"fv"`` referring respectively to `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ and `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_. These 2 methods approximate the pair correlation function of a point process from a realization of the underlying point process using some edge corrections and some basic approximations. For more details see :cite:`Rbook15`. Defaults to ``"fv"``.

            install_spatstat (bool, optional): If ``True``, the `R` package `spatstat <https://github.com/spatstat/spatstat>`_  will be automatically updated or installed (if not present) on your local machine, see also `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_. Note that this requires the installation of the `R programming language <https://cran.r-project.org/>`_ on your local machine.

        Keyword Args:
            params (dict):
                - if ``method = "ppp"``
                    - keyword arguments of `spastat.core.pcf.ppp <https://rdrr.io/cran/spatstat.core/man/pcf.ppp.html>`_,
                - if ``method = "fv"``
                    - Kest = dict(keyword arguments of `spastat.core.Kest <https://rdrr.io/github/spatstat/spatstat.core/man/Kest.html>`_),
                    - fv = dict( keyword arguments of `spastat.core.pcf.fv <https://rdrr.io/cran/spatstat.core/man/pcf.fv.html>`_).

        Returns:
            pandas.DataFrame: Version of the output of `spatstat.core.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.ppp>`_ or `spatsta.core.pcf.fv <https://www.rdocumentation.org/packages/spatstat.core/versions/2.1-2/topics/pcf.fv>`_. The first column of the DataFrame is the set of radius on which the pair correlation function was approximated. The others correspond to the approximated pair correlation function with different edge corrections.

        Example:
            .. literalinclude:: code/pcf_example.py
                :language: python
                :lines: 1-14
                :emphasize-lines: 12-14

        .. proof:definition::

            The pair correlation function of a stationary point process :math:`\mathcal{X}` of intensity :math:`\rho` is the function :math:`g` satisfying (when it exists),

            .. math::

                \mathbb{E} \bigg[ \sum_{\mathbf{x}, \mathbf{y} \in \mathcal{X}}^{\neq}
                f(\mathbf{x}, \mathbf{y}) \bigg] = \int_{\mathbb{R}^d \times \mathbb{R}^d} f(\mathbf{x}+\mathbf{y}, \mathbf{y})\rho^{2} g(\mathbf{x}) \mathrm{d} \mathbf{x} \mathrm{d}\mathbf{y},

            for any non-negative smooth function :math:`f`  with compact support.
        """
        assert point_pattern.dimension in (2, 3)
        assert method in ("ppp", "fv")

        # core, geom and other subpackages are updated if install_spatstat
        spatstat = SpatstatInterface(update=install_spatstat)
        spatstat.import_package("core", "geom", update=False)

        data = point_pattern.convert_to_spatstat_ppp()

        if method == "ppp":
            r = params.get("r", None)
            if r is not None and isinstance(r, np.ndarray):
                params["r"] = robjects.vectors.FloatVector(r)
            pcf = spatstat.core.pcf_ppp(data, **params)

        elif method == "fv":
            params_Kest = params.get("Kest", dict())
            Kest_r = params_Kest.get("r", None)
            if Kest_r is not None and isinstance(Kest_r, np.ndarray):
                params_Kest["r"] = robjects.vectors.FloatVector(Kest_r)

            k_ripley = spatstat.core.Kest(data, **params_Kest)
            params_fv = params.get("fv", dict())
            pcf = spatstat.core.pcf_fv(k_ripley, **params_fv)

        pcf_pd = pd.DataFrame(np.array(pcf).T, columns=pcf.names)
        pcf_pd.drop(columns="theo", inplace=True)
        return pcf_pd

    @staticmethod
    def interpolate(r, pcf_r, clean=True, **params):
        """Interpolate the vector ``pcf_r`` evaluated at ``r``, where NaNs, posinf and neginf of ``pcf_r`` are set to zero if ``clean`` is True.

        The interpolation is performed with `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

        Args:
            r (numpy.ndarray): Vector of radius. Typically, the first colomun of the output of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

            pcf_r (numpy.ndarray): Vector of approximations of the pair correlation function. Typically, a column from the output of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

            clean (bool, optional): Replace nan, posinf and neginf values of ``pcf_r`` by zero before interpolating. Defaults to True.

        Keyword Args:
            params (dict): Keyword arguments of the function `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

        Returns:
            tuple (dict, callable): Dictionary containing the bounds of the support of ``r`` and the resulting output function of the interpolation of ``pcf_r``.

        Example:
            .. literalinclude:: code/sf_baddour_example.py
                :language: python
                :lines: 18-21

        .. note::
            Typically ``pcf_r`` is an approximation of the pair correlation function using the method :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`. The failure of the approximation method on some specific radius may lead to some bad data like nan, posinf and neginf. This may happen for small radiuses, the reason for replacing them with zero. see :cite:`Rbook15`.
        """
        params.setdefault("fill_value", "extrapolate")
        params.setdefault("kind", "cubic")
        if clean:
            pcf_r = utils.set_nan_inf_to_zero(pcf_r)
        pcf = interpolate.interp1d(r, pcf_r, **params)
        dict_r_min_max = dict(r_min=np.min(r), r_max=np.max(r))
        return dict_r_min_max, pcf

    @staticmethod
    def plot(pcf_dataframe, exact_pcf, file_name, **kwargs):
        r"""Plot the columns a DataFrame (excluding the first) with respect to the first columns.

        Args:
            pcf_dataframe (pandas.DataFrame): Output DataFrame of the method :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

            exact_pcf (callable): Function representing the theoretical pair correlation function of the point process.

            file_name (str): Name used to save the figure. The available output formats depend on the backend being used.

            Keyword Args:
                kwargs (dict): Keyword arguments of the function `pandas.DataFrame.plot.line <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html>`_.

        """
        axis = pcf_dataframe.plot.line(x="r", **kwargs)
        if exact_pcf is not None:
            axis.plot(
                pcf_dataframe["r"],
                exact_pcf(pcf_dataframe["r"]),
                "g",
                label="exact pcf",
            )
        utils.plot_poisson(pcf_dataframe["r"], axis=axis, linestyle=(0, (5, 5)))

        axis.legend()
        axis.set_xlabel(r"Radius ($r$)")
        axis.set_ylabel(r"Pair correlation function ($g(r)$)")

        if file_name:
            fig = axis.get_figure()
            fig.savefig(file_name, bbox_inches="tight")
        return axis
