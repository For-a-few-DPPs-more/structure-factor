import numpy as np
import numpy.random as npr

class StructureFactor():
    """
    implement various estimators of the structure factor of a point process.
    """

    def __init__(self, data):
        self.data = np.array(data)
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be an (nxdim) NumPy array")

        try:
            self.N = self.data.shape[0]
            self.d = self.data.shape[1]
        except:
            raise IndexError("Input data should be an (nxdim) NumPy array")

        if self.d != 2:
            raise ValueError("The library only supports 2D patterns so far")

    def get_ensemble_estimate(self, wave_vectors):
        """compute the ensemble estimator described in [Coste, 2020].
        """
        return npr.randn(len(wave_vectors))

    def plot_ensemble_estimate(self, wave_vectors):
        """plot 2D and plot 1D
        """

    def get_fourier_estimate(self, wave_vector_norms):
        """compute the Fourier estimator.
        """
        return npr.randn(len(wave_vectors))
