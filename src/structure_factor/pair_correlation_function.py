"""Collection of functions used to estimate the pair correlation function of an isotropic point process. The underlying routines call the `R` package `spatstat <https://github.com/spatstat/spatstat>`_.

- :py:meth:`~structure_factor.pair_correlation_function.estimate`: Estimates the pair correlation function.

- :py:meth:`~structure_factor.pair_correlation_function.interpolate`: Cleans, interpolates, and extrapolates the results of :py:meth:`~structure_factor.pair_correlation_function.estimate`.

- :py:meth:`~structure_factor.pair_correlation_function.plot`: Plots the results of :py:meth:`~structure_factor.pair_correlation_function.estimate`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
from scipy.interpolate import interp1d
from spatstat_interface.interface import SpatstatInterface

import structure_factor.plotting as plots
import structure_factor.utils as utils


def estimate(point_pattern, method="fv", install_spatstat=False, **params):
    r"""Estimate the pair correlation function (pcf) of a point process encapsulated in ``point_pattern`` (only for stationary isotropic point processes of :math:`\mathbb{R}^2`). The available methods are the methods ``spastat.core.pcf_ppp`` and ``spastat.core.pcf_fv`` of the `R` package `spatstat <https://github.com/spatstat/spatstat>`_.

    .. warning::

        This function requires the `R programming language <https://cran.r-project.org/>`_ to be installed on your local machine but, it doesn't require any knowledge of the programming language R.

    Args:
        point_pattern (:py:class:`~structure_factor.point_pattern.PointPattern`): Realization of the underlying point process.

        method (str, optional): Trigger the use of the routine `pcf.ppp <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.ppp>`_ or `pcf.fv <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.fv>`_ according to the value ``"ppp"`` or ``"fv"``. These 2 methods approximate the pair correlation function of a point process from one realization encapsulated in ``point_pattern``. For more details see :cite:`Rbook15`. Defaults to ``"fv"``.

        install_spatstat (bool, optional): If ``True``, the `R` package `spatstat <https://github.com/spatstat/spatstat>`_  will be automatically updated or installed (if not present) on your local machine, see also `spatstat-interface <https://github.com/For-a-few-DPPs-more/spatstat-interface>`_. Note that this requires the installation of the `R programming language <https://cran.r-project.org/>`_.

    Keyword Args:
        params (dict):

            - if ``method = "ppp"``

                - keyword arguments of `spastat.explore.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.ppp>`_, ex: r, correction ...

            - if ``method = "fv"``

                - **Kest** = dict(keyword arguments of `spastat.explore.Kest <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/Kest>`_), ex: rmax ...

                - **fv** = dict(keyword arguments of `spastat.explore.pcf.fv <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.fv>`_), ex: method, spar ...

    Returns:
        pandas.DataFrame: output of `spastat.explore.pcf.ppp <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.ppp>`_ or `spastat.explore.pcf.fv <https://www.rdocumentation.org/packages/spatstat.explore/versions/3.0-6/topics/pcf.fv>`_. The first column of the DataFrame is the set of radii on which the pair correlation function was approximated. The others correspond to the approximated pair correlation function with different edge corrections.

    Example:
        .. plot:: code/pair_correlation_function/estimate_pcf.py
            :include-source: True

    .. proof:definition::

        The pair correlation function of a stationary isotropic point process :math:`\mathcal{X}` of intensity :math:`\rho` is the function :math:`g` satisfying (when it exists),

        .. math::

            \mathbb{E} \bigg[ \sum_{\mathbf{x}, \mathbf{y} \in \mathcal{X}}^{\neq}
            f(\mathbf{x}, \mathbf{y}) \bigg] = \int_{\mathbb{R}^d \times \mathbb{R}^d} f(\mathbf{x}+\mathbf{y}, \mathbf{y})\rho^{2} g(\mathbf{x}) \mathrm{d} \mathbf{x} \mathrm{d}\mathbf{y},

        for any non-negative smooth function :math:`f` with compact support.

        For more details, we refer to :cite:`HGBLR:22`, (Section 2).

    .. seealso::

        - :py:meth:`~structure_factor.point_pattern.PointPattern`
        - :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFuntcion.interpolate`
        - :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFuntcion.plot`
    """
    assert point_pattern.dimension in (2, 3)
    assert method in ("ppp", "fv")

    # explore, geom and other subpackages are updated if install_spatstat
    spatstat = SpatstatInterface(update=install_spatstat)
    spatstat.import_package("explore", "geom", update=False)

    data = point_pattern.convert_to_spatstat_ppp()

    if method == "ppp":
        r = params.get("r", None)
        if r is not None and isinstance(r, np.ndarray):
            params["r"] = robjects.vectors.FloatVector(r)
        pcf = spatstat.explore.pcf_ppp(data, **params)

    elif method == "fv":
        params_Kest = params.get("Kest", dict())
        Kest_r = params_Kest.get("r", None)
        if Kest_r is not None and isinstance(Kest_r, np.ndarray):
            params_Kest["r"] = robjects.vectors.FloatVector(Kest_r)

        k_ripley = spatstat.explore.Kest(data, **params_Kest)
        params_fv = params.get("fv", dict())
        pcf = spatstat.explore.pcf_fv(k_ripley, **params_fv)

    pcf_pd = pd.DataFrame(np.array(pcf).T, columns=pcf.names)
    pcf_pd.drop(columns="theo", inplace=True)
    return pcf_pd


# todo add test
# todo clean up arguments: only drop, nan, posinf, neginf are necessary, clean and replace can be removed
def interpolate(
    r,
    pcf_r,
    clean=True,
    drop=True,
    replace=False,
    nan=0.0,
    posinf=0.0,
    neginf=0.0,
    extrapolate_with_one=True,
    **params
):
    r"""Interpolate and then extrapolate the evaluation ``pcf_r`` of the pair correlation function (pcf) ``r``.

    The interpolation is performed with `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

    Args:
        r (numpy.ndarray): Vector of radii. Typically, the first column of the output of :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

        pcf_r (numpy.ndarray): Vector of evaluations of the pair correlation function at ``r``. Typically, a column from the output of :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

        clean (bool, optional): If ``True``, a method is chosen to deal with possible outliers (nan, posinf, neginf) of ``pcf_r``. The chosen method depends on the parameters ``replace`` and ``drop``. Defaults to True.

        drop (bool, optional): Cleaning method for ``pcf_r`` active when it's set to True simultaneously with ``clean=True``. Drops possible nan, posinf, and neginf from ``pcf_r`` with the corresponding values of ``r``.

        replace (bool, optional): Cleaning method for ``pcf_r`` active when it's set to True simultaneously with ``clean=True``. Replaces possible nan, posinf, and neginf values of ``pcf_r`` by the values set in the corresponding arguments. Defaults to True.

        nan (float, optional): When ``replace=True``, replacing value of nan present in ``pcf_r``. Defaults to 0.0.

        posinf (float, optional): When ``replace=True``, replacing value of +inf values present in ``pcf_r``. Defaults to 0.0.

        neginf (float, optional): When is ``replace=True``, replacing value of -inf present in ``pcf_r``. Defaults to 0.0.

        extrapolate_with_one (bool, optional): If True, the discrete approximation vector ``pcf_r`` is first interpolated until the maximal value of ``r``, then the extrapolated values are fixed to 1. If False, the extrapolation method of `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ is used. Note that, the approximation of the structure factor by Ogata quadrature Hankel transform is usually better when set to True. Defaults to True.

    Keyword Args:
        params (dict): Keyword arguments of the function `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_.

    Returns:
        callable: Interpolated pair correlation function.

    Example:
        .. plot:: code/pair_correlation_function/interpolate_pcf.py
            :include-source: True

    .. seealso::

        - :py:meth:`~structure_factor.point_pattern.PointPattern`
        - :py:meth:`~structure_factor.pair_correlation_function.PairCorrelationFuntcion.estimate`
    """
    params.setdefault("kind", "cubic")
    if clean:
        if replace:
            pcf_r = utils.set_nan_inf_to_zero(
                pcf_r, nan=nan, posinf=posinf, neginf=neginf
            )
        elif drop:
            index_outlier = np.isnan(pcf_r) | np.isinf(pcf_r)
            pcf_r = pcf_r[~index_outlier]
            r = r[~index_outlier]

    if extrapolate_with_one:
        pcf = lambda x: _extrapolate_pcf(x, r, pcf_r, **params)
    else:
        params.setdefault("fill_value", "extrapolate")
        pcf = interp1d(r, pcf_r, **params)

    return pcf


#! todo add example in the doc
def plot(pcf_dataframe, exact_pcf=None, file_name="", **kwargs):
    r"""Plot the columns of ``pcf_dataframe`` with respect to the column ``pcf_dataframe["r"]``.

    Args:
        pcf_dataframe (pandas.DataFrame): DataFrame to be visualized. Typically the output of :py:meth:`~structure_factor.structure_factor.StructureFactor.estimate`.

        exact_pcf (callable): Theoretical pair correlation function of the point process.

        file_name (str): Name used to save the figure. The available output formats depend on the backend being used.

    Keyword Args:
        kwargs (dict): Keyword arguments of the function `pandas.DataFrame.plot.line <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.line.html>`_.

    Return:
        plt.Axes: Plot result.
    """
    axis = pcf_dataframe.plot.line(x="r", **kwargs)
    if exact_pcf is not None:
        axis.plot(
            pcf_dataframe["r"],
            exact_pcf(pcf_dataframe["r"]),
            "g",
            label=r"Exact $g(r)$",
        )
    plots.plot_poisson(pcf_dataframe["r"], axis=axis, linestyle=(0, (5, 5)))

    axis.legend()
    axis.set_xlabel(r"Radius ($r$)")
    axis.set_ylabel(r"Pair correlation function ($g(r)$)")
    plt.show()
    if file_name:
        fig = axis.get_figure()
        fig.savefig(file_name, bbox_inches="tight")
    return axis


def _extrapolate_pcf(x, r, pcf_r, **params):
    """Interpolate pcf_r for x=<r_max and set to 1 for x>r_max.

    Args:
        x (numpy.ndarray): Points on which the pair correlation function is to be evaluated.

        r (numpy.ndarray): Vector of the radius on with the pair correlation function was evaluated.

        pcf_r (numpy.ndarray): Vector of evaluations of the pair correlation function corresponding to ``r``.

    Returns:
        numpy.ndarray: evaluation of the extrapolated pair correlation function on ``x``.
    """
    r_max = np.max(r)  # maximum radius
    pcf = np.zeros_like(x)
    params.setdefault("fill_value", "extrapolate")

    mask = x > r_max
    if np.any(mask):
        pcf[mask] = 1.0
        np.logical_not(mask, out=mask)
        pcf[mask] = interp1d(r, pcf_r, **params)(x[mask])
    else:
        pcf = interp1d(r, pcf_r, **params)(x)

    return pcf
