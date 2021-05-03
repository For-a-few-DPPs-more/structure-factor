import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from pyhank import HankelTransform
from scipy.integrate import quad
from scipy import interpolate
from mpmath import fp as mpm
from scipy.special import  j0, j1, yv
import pandas as pd
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()

    

class Symmetric_Fourier_Transform():
    """
    implement Symmetric Fourier transform based on OGATA paper "Integration Based On Bessel Function", with a change of variable allowing to 
    approximate the Symmetric Fourier transform, needed to approximate the structure factor of a set of data, by first approximating the pair 
    correlation function (of just having the exact function), and taking the Fourier transform of the total pair correlation function .
    
    input:
        N: int, number of sample points used to approximate the integral by a sum.
        h: float, step size in the sum.
        r_vector: array, vector containing the radius on which the pair correlation function is evaluated
        g: function, pair correlation function if it's  known else it will be approximated (see class Structure_Factor)
        data_g: array_like(r_vector), vector containing the evaluations of the pair correlation function on r_vec
        k: array, vector containing the wavelength on which we want to approximate the structure factor
    output:
        ret: array_like(k), estimation of the fourier transform of the total correlation function (pair correlation function -1)
        k_min: float, minimum confidence value of wavelength  
    """

    def __init__(self, N=None, h=0.1):
        
        self._h = h
        if not isinstance(N, int):
            raise TypeError("N should be an integer.")
        
        def roots(N):
            return np.array([mpm.besseljzero(0, i + 1) for i in range(N)]) / np.pi #first N Roots of the Bessel J_0 functions divided by pi.
        self._zeros = roots(N) # Xi
       
        def psi(t):
            return t * np.tanh(np.pi * np.sinh(t) / 2)

        def get_x(h, zeros):
            return np.pi * psi(h * zeros) / h 
        self.x = get_x(h, self._zeros) #pi*psi(h*ksi/pi)/h
        self.kernel = j0(self.x) #J_0(pi*psi(h*ksi))

        def weight(zeros):
            return yv(0, np.pi * zeros) / j1(np.pi * zeros)
        self.w = weight(self._zeros) #(Y_0(pi*zeros)/J_1(pi*zeros))
        
        def d_psi(t):
            t = np.array(t, dtype=float)
            d_psi = np.ones_like(t)
            exact_t = t < 6
            t = t[exact_t]
            d_psi[exact_t] = (np.pi * t * np.cosh(t) + np.sinh(np.pi * np.sinh(t))) / (
                1.0 + np.cosh(np.pi * np.sinh(t))
            )
            return d_psi
        self.dpsi = d_psi(h * self._zeros) #dpsi(h*ksi)        
        self._factor = None
    
    def _f(self, r_vector, data_g):
        data_f = data_g - 1 
        self.f = interpolate.interp1d(r_vector, data_f, axis=0, fill_value='extrapolate', kind='cubic')
        return(self.f)

    def _k(self, k):
        return np.array( np.array(k))

    @property
    def _series_fac(self):
        if self._factor is None:
            self._factor = np.pi * self.w * self.kernel * self.dpsi #pi*w*J_0(pi*psi(h*ksi))*dpsi(h*ksi)
        return self._factor

    def _get_series(self, f, k, alpha):
        with np.errstate(divide="ignore"):  # numpy safely divides by 0
            args = np.divide.outer(self.x, k).T  # x/k
        return self._series_fac * (f(args) -1*alpha) * (self.x)    

    def transform(self, k, g=None,  r_vector=None, data_g=None ):
        k = self._k(k) 
        if g == None:
            f = self._f(r_vector, data_g)
            self.k_min = (np.pi * 3.2)/(self._h* np.max(r_vector))
            summation = self._get_series(f, k, alpha=0) # pi*w*J0(x)
            
        else :
            self.k_min = np.min(k)
            summation = self._get_series(g, k, alpha=1) # pi*w*J0(x)
           
        ret = np.empty(k.shape, dtype=summation.dtype)
        ret = np.array(2*np.pi * np.sum(summation, axis=-1) / np.array(k **2)) #2pi*summation/k**2
        return (ret, self.k_min)


class Structure_Factor(Symmetric_Fourier_Transform):
    """
    implement various estimators of the structure factor of a point process in dimension 2.
    
    input:
        data : 2d_array, data set of shape nxdim NumPy array (x_data and y_data are the coordinates of data's points)
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

    def get_scattering_intensity_estimate(self, L, max_wave_lenght, arg="1D"):
        """compute the ensemble estimator described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5) 
        which is an approximation of the structure factor, but at zero it gives different result.
        the data should be simulated in a square of length L 
        L : int: length of the square that contains the data  
        max_wave_length : int  maximum wavelength
        arg: str: 1D, 2D. 
        si_ : the sum in the formula of the scattering intensity
        si : scattering intensity of data for wave vectors defined by (x_waves, y_waves)
        """
        x_max = np.floor(max_wave_lengh*L/(2*np.pi*np.sqrt(2)))
        if arg=="2D":
            x_grid = np.linspace(1, x_max, x_max)
            x_waves, y_waves = np.meshgrid(x_grid, x_grid)
        else:
            x_waves = np.linspace(1, x_max, x_max)
            y_waves = x_waves
        self.x_waves = x_waves
        self.y_waves = y_waves
        #if x_waves.shape != y_waves.shape :
         #   raise IndexError("x_waves and y_waves should have the same shape.")
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
            pcf_estimation = pcf(data_r, spar=0.4, correction=correction_)
                             
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
            pcf_estimation = robjects.conversion.rpy2py(robjects.r('pcf.fv(kest_data, spar=0.4, all.knots=FALSE,  keep.data=TRUE,  method=method_)'))  
                             
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
        
    def get_fourier_estimate(self, args, arg_2, N= None, h=0.1, wave_lengh=None): 
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
            super().__init__(N=N, h=h)

            #h_estimation_interpolate = interpolate.interp1d(r_vec, h_estimation, axis=0, fill_value='extrapolate', kind='cubic')
            sf, self.k_min = super().transform(data_g=g_to_plot,r_vector=r_vec, k=wave_lengh)
            print("The fialble minimum wavelenght is :",  self.k_min)
            sf_estimation_2 = 1 + intensity * sf
            sf_interpolate = interpolate.interp1d(wave_lengh, sf_estimation_2, axis=0, fill_value='extrapolate', kind='cubic')
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
            ax[1].plot(self.k_min, sf_interpolate(self.k_min), "ro", label="fiable k_min")  
            ax[1].plot(wave_lengh, ones_, 'r--', label="y=1")
            ax[1].legend()
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('S(k)')
            ax[1].title.set_text('structur factor of data' )
            plt.show()
            return( wave_lengh, sf_estimation_2, self.k_min)
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
