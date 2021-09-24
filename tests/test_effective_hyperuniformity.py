import numpy as np
import pytest

from hypton.effective_hyperuniform import EffectiveHyperuniformity
from hypton.utils import structure_factor_ginibre, structure_factor_poisson

k = np.linspace(0, 10, 100)


@pytest.mark.parametrize(
    "k, sf_k, expected",
    [
        (k, structure_factor_poisson(k), False),
        (k, structure_factor_ginibre(k), True),
    ],
)
# ? I don't understand what is tested: the value of the hyperuniformity index or the fact that is it small (< 1e-3)
# todo use a more explicit name, test_small_hyperuniformity_index or test_hyperuniformity_index_should_be...
def test_hyperuniformity_index(k, sf_k, expected):
    # todo rename hyp_test, it reads like hypothesis test
    hyp_test = EffectiveHyperuniformity(k, sf_k)
    index_H, _ = hyp_test.index_H(norm_k_stop=4)
    result = index_H < 1e-3
    assert result == expected
