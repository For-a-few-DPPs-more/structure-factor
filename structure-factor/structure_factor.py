import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from pyhank import HankelTransform
from scipy.integrate import quad
from scipy import interpolate
from mpmath import fp as mpm
from scipy.special import gamma, j0, j1, jn, jv, yv
from scipy.special import jn_zeros as _jn_zeros
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

    
class Symmetric_Fourier_Transform():

    def __init__(self, ndim=2, N=None, h=0.05):
        self._h = h
        self.ndim = ndim
    
        def roots(N):
            return np.array([mpm.besseljzero(0, i + 1) for i in range(N)]) / np.pi #first N Roots of the Bessel J(nu) functions divided by pi.
        self._zeros = roots(N)
       
        def psi(t):
            return t * np.tanh(np.pi * np.sinh(t) / 2)

        def get_x(h, zeros):
            return np.pi * psi(h * zeros) / h 
        self.x = get_x(h, self._zeros) #pi*psi(h*ksi/pi)
    
        def kernel(x, nu):
            if np.isclose(nu, 0):
                return j0(x)
            if np.isclose(nu, 1):
                return j1(x)
            if np.isclose(nu, np.floor(nu)):
                return jn(int(nu), x)
            return jv(nu, x)
        self.kernel = kernel(self.x, 0) #J_0(pi*psi(h*ksi))

        def weight(zeros):
            return yv(0, np.pi * zeros) / kernel(np.pi * zeros, 1)
        self.w = weight(self._zeros) #(Y_0(pi*zeros)/J_1(pi*zeros))
        
        def d_psi(t):
            t = np.array(t, dtype=float)
            a = np.ones_like(t)
            mask = t < 6
            t = t[mask]
            a[mask] = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (
                1.0 + np.cosh(np.pi * np.sinh(t))
            )
            return a
        self.dpsi = d_psi(h * self._zeros) #dpsi(h*ksi)        
        self._factor = None

    def _k(self, k):
        return np.array( np.array(k))

    @property
    def _series_fac(self):
        if self._factor is None:
            self._factor = np.pi * self.w * self.kernel * self.dpsi #pi*w*J_0(pi*psi(h*ksi))*dpsi(h*ksi)
        return self._factor

    def _get_series(self, f, k=1):
        with np.errstate(divide="ignore"):  # numpy safely divides by 0
            args = np.divide.outer(self.x, k).T  # x = r*k
        return self._series_fac * f(args) * (self.x)    

    def transform(self, f, k=1):
        k = self._k(k) # k as array
        #k_0 = np.isclose(k, 0) #index for zeros k
        #kn0 = np.invert(k_0) # index  for non zero k
        #k_tmp = k[kn0] # kwithout values close to zero
        k_tmp = k
        knorm = np.array(k_tmp **2) #k**2
        # The basic transform has a norm of 1.
        norm = (2 * np.pi) 
        summation = self._get_series(f, k_tmp) # pi*w*J0(x)
        ret = np.empty(k.shape, dtype=summation.dtype)
        #ret[kn0] = np.array(norm * np.sum(summation, axis=-1) / knorm) #2pi*summation/k**2
        ret = np.array(norm * np.sum(summation, axis=-1) / knorm) #2pi*summation/k**2
        # care about k=0
        #ret_0 = 0
        #if np.any(k_0):

            #def integrand(r):
                #return f(r).real * (r)

            #int_res = quad(integrand, 0, np.inf)
            #ret_0 = int_res[0] * norm
            #ret[k_0] = ret_0
        return ret


class StructureFactor(Symmetric_Fourier_Transform):
    """
    implement various estimators of the structure factor of a point process.
    data : data set of shape nxdim NumPy array (x_data and y_data are the coordinates of data's points)
    n_data :  number of points of in data
    intensity: average number of data in unit volume
    """

    def __init__(self, data, intensity: float):
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be an (nxdim) NumPy array")

        try:
            self.n_data = data.shape[0]
            self.d = data.shape[1]
        except:
            raise IndexError("Input data should be an (nxdim) NumPy array")

        if self.d != 2:
            raise ValueError("The library only supports 2D patterns so far")

        
        self.data = np.array(data)
        self.intensity = intensity
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]
        #self.n_data = max(self.x_data.shape)

    def get_scattering_intensity_estimate(self, x_waves, y_waves):
        """compute the ensemble estimator described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5) 
        which is an approximation of the structure factor, but at zero it gives different result.
        x_waves : x coordinates of the wave vector 
        y_waves : y coordinates of the wave vector
        si_ : the sum inthe formula of the scattering intensity
        si : scattering intensity of data for wave vectors defined by (x_waves, y_waves)
        """
        self.x_waves = x_waves
        self.y_waves = y_waves
        if x_waves.shape != y_waves.shape :
            raise IndexError("x_waves and y_waves should have the same shape.")
        si_ = 0 # initial value of the sum in the scattering intensity
        x_data = self.x_data 
        y_data = self.y_data 
        n_data = self.n_data
        for k in range(0,n_data):
            si_ = si_ + np.exp(- 1j * (x_waves * x_data[k] + y_waves * y_data[k]))
        self.si = (1 / n_data)*np.abs(si_) ** 2
        return self.si

    def plot_scattering_intensity_estimate(self, arg):
        """plot 2D and plot 1D
        wave_lengh : norm of the waves defined by (x_waves, y_waves)
        arg : (str) could be "all", "color_level" or "plot".
             define the type of plot to be visualized
        x_ones_, ones_ : vectors implemented to add the line y=1 to the plot whic correspond to the 
                        theoretical value of the structure factor of a Poisson point process
        """
        x_waves = self.x_waves
        y_waves = self.y_waves
        si = self.si
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
                ax[1].legend(["Scattering intensity", "y=1" ], shadow=True,loc=1)
                ax[1].set_xlabel("wave lengh (k)")
                ax[1].set_ylabel("scattering intensity (SI(k))")
                ax[1].title.set_text("loglog plot")
                f_0 = ax[2].imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
                fig.colorbar(f_0, ax = ax[2])
                ax[2].title.set_text("scattering intensity")
                plt.show()
        elif arg == "plot":
            plt.loglog(wave_lengh, si, 'k,')
            plt.loglog(x_ones_, ones_, 'r--')
            plt.legend(['Scattering intensity','y=1'], loc=1)
            plt.xlabel("wave lengh (k)")
            plt.ylabel("Scattering intensity (SI(k))")
            plt.title("loglog plot")
            plt.show()
        elif arg == "color_level":
            if np.min(x_waves.shape) == 1 or np.min(y_waves.shape) == 1 :
                
                raise ValueError("X_waves, Y_waves should be meshgrids or choose arg = 'plot'. ")
            else :
                f_0 = plt.imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2. ], cmap="PRGn")
                plt.colorbar(f_0)
                plt.title("Scattering intensity")
                plt.show()
        else :
            raise ValueError("arg should be one of the following str: 'all', 'plot' and 'color_level'.  ")
    def get_pcf_estimate(self, raduis, args, correction_=None, r_vec=None, r_max=None):
        """compute the pair correlation function of data using the R packadge spatstat pcf.ppp, and pcf. fv
        args : (srt) should be 'fv' or 'ppp'. If it is set to be 'fv' then pcf.fv is used, if 'ppp' then 
              pcf.ppp is used. 
        raduis: raduis of the ball which contains the data on which the pair correlation function will
                be conmputed.
        correction_: if args= 'ppp' : correction ( should be one of : "translate", "Ripley", "isotropic",
                    "best", "good" , "all"
                    if args='fv' : keep the default value 'None'.
        r_vec : if args = 'ppp' : keep the default 'None'
                if arg = 'fv' : is the vector of raduis to evaluate g. it's prefered to can keep the default 
                or to set an r_max 
        r_max : if args = 'ppp' : keep the default 'None'
                if arg = 'fv' : r_max is the maximum raduis on which g will be evaluated
        g: the pair correlation function 
        g_pd: the pair coorelation function as data frame
        x_data_r: x_data transfered into R object 
        y_data_r: y_data transfered into R object
        data_r : data transformed into R object
        
        for more details see : "https://rdrr.io/cran/spatstat/man/pcf.ppp.html")
              if arg_1 = "fv": arg_2 is the method ("a", "b", "c" or "d")
        """
        if args not in ["fv", "ppp"]:
            raise ValueError(" args should be 'fv', or 'ppp'")
        if not np.isscalar(raduis):
            raise ValueError("raduis must be a scalar")
        
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind = 1)
        spatstat = rpackages.importr('spatstat')
        disc = robjects.r("disc")
        center = robjects.r('c')
        r_base = importr('base')
        ppp = robjects.r('ppp')
        pcf = robjects.r('pcf')
        #pcf_fv = robjects.r('pcf.fv')
        Kest = robjects.r('Kest')
        #r_hist = robjects.r("hist")
        #barplot = robjects.r("barplot")
        #owin = robjects.r("owin") #si on prend un rectangle

        x_data = self.x_data 
        y_data = self.y_data 


        x_data_r = robjects.vectors.FloatVector(x_data)
        y_data_r = robjects.vectors.FloatVector(y_data)
        
        data_r = ppp(x_data_r, y_data_r, window=disc(raduis, center(0, 0)))                    
        if args == "ppp":          
            if correction_ not in ["translate", "Ripley", "isotropic", "best", "good" , "all", None] :
                raise ValueError("correction should be one of the following str: 'translate', 'Ripley', 'isotropic', 'best', 'good' , 'all', 'none'.")
            pcf_estimation = pcf(data_r, correction=correction_)
                             
        if args == "fv":
            if correction_ not in ["a", "b", "c", "d", None] :
                raise ValueError("correction_ should be one of the following str: 'a', 'b', 'c', 'd', None.")  
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
            robjects.globalenv['method_']= correction_
            pcf_estimation = robjects.conversion.rpy2py(robjects.r('pcf.fv(kest_data,  method=method_)'))  
                             
        self.pcf_estimation_pd = pd.DataFrame.from_records(pcf_estimation).fillna(0) # as pandas data frame
        return self.pcf_estimation_pd
    
    def plot_pcf_estimate(self, args):
        """
        plot the pair correlation function estimation using get_pcf_estimate. 
        args: (str), should be 'pcf', 'trans', 'iso' or 'un'
        """
        pcf_estimation_pd = self.pcf_estimation_pd
        g_key = (pcf_estimation_pd.keys()).tolist()
        if g_key.count(args) == 0:
            raise ValueError("The data frame does not contains the chosen args. Check pcf_estimation_pd.keys() to plot one of them. ")
        r_vec = np.array(pcf_estimation_pd["r"])
        g_to_plot = np.array(pcf_estimation_pd[args])
        g_theo = np.array(pcf_estimation_pd["theo"])
        plt.plot(r_vec, g_theo, 'r--', label="y=1")
        plt.scatter(r_vec, g_to_plot, c='k', s=1, label="pcf")
        plt.plot(r_vec, g_to_plot, 'b')
        plt.legend()
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Pair correlation function ")
        
    def get_fourier_estimate(self, args, arg_2, N= None, h=None, wave_lengh=None): 
        """
         compute the approximation of the structure factor of data by evaluating the fourier transform of the paire
        approximated by the R packadge spatstat
        args: (str) one of the following str: 'pcf', 'ppp_trans', 'ppp_iso' or 'ppp_un'. It specified the method chosen
        to approximate the pcf 
        intensity: intensity of the point process
        'pcf' if pcf.fv is used
        'trans', 'iso' or 'un': if pcf.ppp is used and trans, iso, un to specifiy which edge correction is used
        arg_2: estimation_1, estimation_2
              estimation_1:  estimation of hankel transform via Quasi hankel transform Vega and Sicairos 2004.
              estimation_2 : using  symetric_fourrier_transform based on ogata 2005 with a change of variable 
                            h = step, 
                            N= number of point used to approximate the integral by a sum 
                            wave_lengh = vector contaning the wave vectors k on which we evaluate the strcture factor.
        """
        intensity = self.intensity
        pcf_estimation_pd = self.pcf_estimation_pd
        g_key = (pcf_estimation_pd.keys()).tolist()
        if not np.isscalar(intensity):
            raise ValueError("intensity must be a scalar")
        if g_key.count(args) == 0:
                raise ValueError("The data frame does not contain the chosen args. Check pcf_estimation_pd.keys() to plot one of them. ")
        g_to_plot = np.array(pcf_estimation_pd[args])
        r_vec = np.array(pcf_estimation_pd["r"])
        g_theo = np.array(pcf_estimation_pd["theo"])
        h_estimation = pcf_estimation_pd[args] -1
        if arg_2 == "estimation_2":
            if N==None :
                N = 1000
            if h ==None :
                h=0.0001
            super().__init__(N=N, h=h)
            h_estimation_interpolate = interpolate.interp1d(r_vec, h_estimation, axis=0, fill_value='extrapolate', kind='cubic')
            #h_estimation_interpolate = lambda x : - np.exp(-x**2)
            sf_estimation_2 = 1 + intensity * super().transform(h_estimation_interpolate, wave_lengh)
            ones_ = np.ones(sf_estimation_2.shape).T
            fig , ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, 'r--', label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c='k', s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, 'b')
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[0].title.set_text("Pair correlation function ")
            ax[1].plot(wave_lengh[1:], sf_estimation_2[1:],'b' )
            ax[1].scatter(wave_lengh, sf_estimation_2, c='k', s=1, label="sf")  
            ax[1].plot(wave_lengh, ones_, 'r--', label="y=1")
            ax[1].legend()
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('S(k)')
            ax[1].title.set_text('structur factor of data' )
            plt.show()
            return( wave_lengh, sf_estimation_2)
        if arg_2 == "estimation_1":
            h_estimation[0] = -1 
            transformer = HankelTransform(order=0, max_radius=max(pcf_estimation_pd['r']), n_points=pcf_estimation_pd['r'].shape[0])
            sf_estimation = 1 + intensity*transformer.qdht(h_estimation)
            wave_lengh = transformer.kr
            ones_ = np.ones(sf_estimation.shape).T
            fig , ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, 'r--', label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c='k', s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, 'b')
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[0].title.set_text("Pair correlation function ")
            ax[1].plot(wave_lengh[1:], sf_estimation[1:],'b' )
            ax[1].scatter(wave_lengh, sf_estimation, c='k', s=1, label="sf")  
            ax[1].plot(wave_lengh, ones_, 'r--', label="y=1")
            ax[1].legend()
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('S(k)')
            ax[1].title.set_text('structur factor of data' )
            plt.show()
            return (wave_lengh, sf_estimation)
