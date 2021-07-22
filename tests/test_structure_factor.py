import pytest

import numpy as np
import numpy.random as npr

import hypton


def test_validity_of_input():
    """
    make sure we run into an error if the pattern is not 2D
    """
    with pytest.raises(TypeError):
        # Try to input a list instead of an array
        point_pattern = zip(npr.rand(10), npr.rand(10))
        sf = hypton.StructureFactor(point_pattern)

    with pytest.raises(IndexError):
        # Try to input an array of the wrong dimension
        point_pattern = np.array(npr.rand(10))
        sf = hypton.StructureFactor(point_pattern)


def test_ensemble_estimate():
    """
    make sure the ensemble estimate discussed in (Coste, 2020) runs as expected
    """
    point_pattern = npr.rand(100).reshape((50, 2))
    sf = hypton.StructureFactor(point_pattern)
    wave_vectors = [npr.randn(2) for _ in range(10)]
    result = sf.get_ensemble_estimate(wave_vectors)
    assert len(result) == len(wave_vectors)
    assert (result >= 0).all()
    assert (result <= 1).all()
