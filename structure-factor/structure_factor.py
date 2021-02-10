import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import pandas as pd
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.conversion import localconverter
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

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
        if x_waves.shape != y_waves.shape :
            raise IndexError("x_waves and y_waves should have the same shape.")
        si = 0 # initial value of scatteing intensity
        x_data = self.x_data #reshape x coordinate's of the point process into the well shape
        y_data = self.y_data #reshape y coordinate's of the point process into the well shape
        n_data = self.n_data
        for k in range(0,n_data):
            si = si + np.exp(- 1j * (x_waves * x_data[k] + y_waves * y_data[k]))
        si = (1 / n_data)*np.abs(si) ** 2
        return si

    def plot_ensemble_estimate(self, arg, si, x_waves, y_waves):
        """plot 2D and plot 1D
        wave_lengh : norm of the waves defined by (x_waves, y_waves)
        arg : could be "all", "color_level", "plot" 
        """
        wave_lengh =  np.sqrt(np.abs(x_waves)**2 + np.abs(y_waves)**2 )
        ones_ = np.ones((x_waves.shape)).T
        x_ones_ = np.linspace(np.min(wave_lengh), np.max(wave_lengh), np.max(x_waves.shape))
        if arg == "all":
            if np.min(x_waves.shape) == 1 or np.min(y_waves.shape) == 1 :
                raise ValueError("X_waves, Y_waves should be meshgrids or choose arg = 'plot'. ")
            else: 
                fig , ax = plt.subplots(1, 3, figsize=(24, 7))
                ax[0].plot(self.x_data, self.y_data, 'b.')
                ax[0].title.set_text("data")
                ax[1].loglog(wave_lengh, si, 'k,')
                ax[1].loglog(x_ones_, ones_, 'r--')
                ax[1].legend(['y=1'], shadow=True,loc=1)
                ax[1].set_xlabel("wave lengh")
                ax[1].set_ylabel("scattering intensity (SI)")
                ax[1].title.set_text("loglog plot")
                f_0 = ax[2].imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
                fig.colorbar(f_0, ax = ax[2])
                ax[2].title.set_text("scattering intensity")
                plt.show()
        elif arg == "plot":
            plt.loglog(wave_lengh, si, 'k,')
            plt.loglog(x_ones_, ones_, 'r--')
            plt.legend(['y=1'], loc=1)
            plt.xlabel("wave lengh")
            plt.ylabel("scattering intensity (SI)")
            plt.title("loglog plot")
            plt.show()
        elif arg == "color_level":
            if np.min(x_waves.shape) == 1 or np.min(y_waves.shape) == 1 :
                raise ValueError("X_waves, Y_waves should be meshgrids or choose arg = 'plot'. ")
            else :
                f_0 = plt.imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
                plt.colorbar(f_0)
                plt.title("scattering intensity")
                plt.show()
        else :
            raise ValueError("arg should be one of the following str: 'all', 'plot' and 'color_level'.  ")
    def get_pcf_ppp_estimate(self, raduis, args):
        """compute the pair correlation function of data using the R packadge spatstat pcf.ppp
        raduis: raduis of the ball which contains the data on which the pair correlation function will be conputed
        g: the pair correlation function 
        g_pd: the pair coorelation function as data frame
        x_data_r: x_data transered into R object 
        y_data_r: y_data transfered into R object
        data_r : data transformed into R object
        r_vec : default  
        args: correction ( should be one of : "translate", "Ripley", "isotropic", "best", "good" , "all", "none"
        for more details see : "https://rdrr.io/cran/spatstat/man/pcf.ppp.html")
        """
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind = 1)
        spatstat = rpackages.importr('spatstat')
        ppp = robjects.r('ppp')
        pcf = robjects.r('pcf')
        disc = robjects.r("disc")
        center = robjects.r('c')
        r_plot = robjects.r('plot')
        r_base = importr('base')
        r_hist = robjects.r("hist")
        barplot = robjects.r("barplot")
        owin = robjects.r("owin")
        
        x_data = self.x_data 
        y_data = self.y_data 
        n_data = self.n_data
        
        x_data_r = robjects.vectors.FloatVector(x_data)
        y_data_r = robjects.vectors.FloatVector(y_data)
        data_r = ppp(x_data_r, y_data_r, window=disc(raduis, center(0, 0)))

        g = pcf(data_r, correction=args)
        g_pd = pd.DataFrame.from_records(g) # as pandas data frame
        return (g_pd)
    def get_pcf_fv_estimate(self, raduis, method_, r_vec=None, r_max = None):
        """compute the pair correlation function of data using the R packadge spatstat pcf.ppp
        raduis: raduis of the ball which contains the data on which the pair correlation function will be conputed
        g: the pair correlation function 
        g_pd: the pair coorelation function as data frame
        x_data_r: x_data transered into R object 
        y_data_r: y_data transfered into R object
        data_r : data transformed into R object
        r_vec : r_vec that we wnat to evaluate pcf on it (default is a sensible vector provided by the R packadge) 
        method: "a", "b", "c" or "d". 
        for more details see : "https://rdrr.io/cran/spatstat/man/pcf.ppp.html")
        """
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind = 1)
        spatstat = rpackages.importr('spatstat')
        ppp = robjects.r('ppp')
        pcf = robjects.r('pcf')
        pcf_fv = robjects.r('pcf.fv')
        Kest = robjects.r('Kest')
        disc = robjects.r("disc")
        center = robjects.r('c')
        r_plot = robjects.r('plot')
        r_base = importr('base')
        r_hist = robjects.r("hist")
        barplot = robjects.r("barplot")
        owin = robjects.r("owin")
        
        x_data = self.x_data 
        y_data = self.y_data 
        n_data = self.n_data
        
        x_data_r = robjects.vectors.FloatVector(x_data)
        y_data_r = robjects.vectors.FloatVector(y_data)
        data_r = ppp(x_data_r, y_data_r, window=disc(raduis, center(0, 0)))

        robjects.globalenv['data_r'] = data_r #Transfert ppp_data comme variable dans R.
        if r_max != None :
            robjects.globalenv['r_max']= r_max
            robjects.r('kest_data = Kest(data_r, rmax=r_max)')
        elif r_vec is not None:
            robjects.globalenv['r_vec']= r_vec
            r_vec_r = robjects.vectors.FloatVector(r_vec)
            robjects.r('kest_data = Kest(data_r, r=r_vec)')
        else :
            robjects.r('kest_data = Kest(data_r)')
        robjects.globalenv['method_']= method_
        g = robjects.conversion.rpy2py(robjects.r('pcf.fv(kest_data,  method=method_)')) 
        g_pd = pd.DataFrame.from_records(g) # as pandas data frame
        return (g_pd)
    
    def plot_pcf_estimate(self, g_pd, args):
        g_key = (g_pd.keys()).tolist()
        if g_key.count(args) == 0:
            raise ValueError("The data frame does not contain the chosen args. Check g.keys() to plot one of them. ")
        r_vec = np.array(g_pd["r"])
        g_to_plot = np.array(g_pd[args])
        g_theo = np.array(g_pd["theo"])
        plt.plot(r_vec, g_theo, 'r--', label="y=1")
        plt.plot(r_vec, g_to_plot, 'k.')
        plt.legend(loc=1)
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Pair correlation function of data")
    def get_fourier_estimate(self, wave_vector_norms):
        """compute the Fourier estimator.
        """
        return npr.randn(len(wave_vectors))



