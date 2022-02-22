import numpy as np
import pytest

import structure_factor.pair_correlation_function as pcf
from structure_factor.data import load_data
from structure_factor.point_processes import GinibrePointProcess


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


# ? why changing the radius values in the first test but not the other?


def test_default_pcf_interpolant_on_ginibre_data():
    r = np.linspace(0, 80, 500)
    pcf_r = GinibrePointProcess.pair_correlation_function(r)
    interp_pcf = pcf.interpolate(r, pcf_r)

    r = np.linspace(5, 10, 30)
    computed_pcf = interp_pcf(r)
    expected_pcf = GinibrePointProcess.pair_correlation_function(r)
    np.testing.assert_almost_equal(computed_pcf, expected_pcf)


def test_pcf_estimation_on_ginibre_data(ginibre_pp):
    param_Kest = dict(r_max=45)
    param_fv = dict(method="b", spar=0.1)
    pcf_fv = pcf.estimate(ginibre_pp, method="fv", Kest=param_Kest, fv=param_fv)

    r, pcf_r = pcf_fv["r"], pcf_fv["pcf"]
    pcf_fv_func = pcf.interpolate(r=r, pcf_r=pcf_r, clean=True)

    computed_pcf = pcf_fv_func(r)
    expected_pcf = GinibrePointProcess.pair_correlation_function(r)
    np.testing.assert_almost_equal(
        computed_pcf,
        expected_pcf,
        decimal=1,
    )
