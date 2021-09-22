import numpy as np
import pytest
from hypton.effective_hyperuniform import EffectiveHyperuniformity


def sf_ginibre(x):
    return 1 - np.exp(-(x ** 2) / 4)


k = np.linspace(0, 10, 100)


@pytest.mark.parametrize(
    "norm_k, sf, expected",
    [
        (k, np.ones_like(k), False),
        (k, sf_ginibre(k), True),
    ],
)
def test_H_index(norm_k, sf, expected):
    hyp_test = EffectiveHyperuniformity(norm_k, sf)
    index_H, _ = hyp_test.index_H(norm_k_stop=4)
    result = index_H < 1e-3
    assert result == expected
