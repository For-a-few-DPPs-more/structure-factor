import numpy as np
import pytest

from structure_factor.data import load_data
from structure_factor.pair_correlation_function import PairCorrelationFunction as PCF
from structure_factor.utils import pair_correlation_function_ginibre


@pytest.fixture
def ginibre_pp():
    return load_data.load_ginibre()


# ? why changing the radius values in the first test but not the other?


def test_default_pcf_interpolant_on_ginibre_data(ginibre_pp):
    r = np.linspace(0, 80, 500)
    pcf_r = pair_correlation_function_ginibre(r)
    _, interp_pcf = PCF.interpolate(r, pcf_r)

    r = np.linspace(5, 10, 30)
    computed_pcf = interp_pcf(r)
    expected_pcf = pair_correlation_function_ginibre(r)
    np.testing.assert_almost_equal(computed_pcf, expected_pcf)


def test_pcf_estimation_on_ginibre_data(ginibre_pp):
    param_Kest = dict(r_max=45)
    param_fv = dict(method="b", spar=0.1)
    pcf_fv = PCF.estimate(ginibre_pp, method="fv", Kest=param_Kest, fv=param_fv)

    r, pcf_r = pcf_fv["r"], pcf_fv["pcf"]
    _, pcf_fv_func = PCF.interpolate(r=r, pcf_r=pcf_r, clean=True)

    computed_pcf = pcf_fv_func(r)
    expected_pcf = pair_correlation_function_ginibre(r)
    np.testing.assert_almost_equal(
        computed_pcf,
        expected_pcf,
        decimal=1,
    )
