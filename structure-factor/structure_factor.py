import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

class StructureFactor():
    """
    implement various estimators of the structure factor of a point process.
    data : data set of shape nxdim NumPy array (x_data and y_data are the coordinates of data's points)
    n_data :  number of points of in data
    """

    def __init__(self, data):
        self.data = np.array(data)
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]
        self.n_data = max(self.x_data.shape)
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be an (nxdim) NumPy array")

        try:
            self.N = self.data.shape[0]
            self.d = self.data.shape[1]
        except:
            raise IndexError("Input data should be an (nxdim) NumPy array")

        if self.d != 2:
            raise ValueError("The library only supports 2D patterns so far")

    def get_ensemble_estimate(self, x_waves, y_waves):
        """compute the ensemble estimator described in [Coste, 2020].
        x_waves : x coordinates of the wave vector 
        y_waves : y coordinates of the wave vector
        si : scattering intensity of data for wave vectors defined by (x_waves, y_waves)
        """
        si = 0 # initial value of scatteing intensity
        x_data = self.x_data #reshape x coordinate's of the point process into the well shape
        y_data = self.y_data #reshape y coordinate's of the point process into the well shape
        n_data = self.n_data
        for k in range(0,n_data):
            si = si + np.exp(- 1j * (x_waves * x_data[k] + y_waves * y_data[k]))
        si = (1 / n_data)*np.abs(si) ** 2;
        return si

    def plot_ensemble_estimate(self, arg, si, x_waves, y_waves):
        """plot 2D and plot 1D
        wave_lengh : norm of the waves defined by (x_waves, y_waves)
        arg : could be "all", "color_level", "plot" 
        """
        wave_lengh =  np.sqrt(np.abs(x_waves)**2 + np.abs(y_waves)**2 )
        ones_ = np.ones((x_wave.shape)).T
        x_ones_ = np.linspace(np.min(wave_lengh), np.max(wave_lengh), np.max(x_wave.shape))
        if arg == "all" :
            fig , ax = plt.subplots(1, 3, figsize=(24, 7))
            ax[0].plot(self.x_data, self.y_data, 'b.')
            ax[0].title.set_text("data")
            ax[1].loglog(wave_lengh, si, 'k,')
            ax[1].loglog(x_ones_, ones_, 'r-', label="y=1")
            ax[1].legend(loc=1)
            ax[1].set_xlabel("wave lengh")
            ax[1].set_ylabel("scattering intensity (SI)")
            ax[1].title.set_text("loglog plot")
            f_0 = ax[2].imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
            fig.colorbar(f_0, ax = ax[2])
            ax[2].title.set_text("scattering intensity")
        elif arg == "plot":
            plt.loglog(wave_lengh, si, 'k,')
            plt.loglog(x_ones_, ones_, 'r-', label="y=1")
            plt.legend(loc=1)
            plt.xlabel("wave lengh")
            plt.ylabel("scattering intensity (SI)")
            plt.title("loglog plot")
            plt.show()
        elif arg == "color_level":
            f_0 = plt.imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
            plt.colorbar(f_0)
            plt.title("scattering intensity")
            plt.show()
        else :
            raise ValueError("arg should be one of the following str 'all', 'plot' and 'color_level'.  ")
    def get_fourier_estimate(self, wave_vector_norms):
        """compute the Fourier estimator.
        """
        return npr.randn(len(wave_vectors))

