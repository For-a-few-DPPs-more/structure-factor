from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from structure_factor.utils import roots, psi, d_psi, get_x, weight
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from pyhank import HankelTransform
from scipy.integrate import quad
from scipy import interpolate
from scipy.special import jv
import pandas as pd
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()


class SymmetricFourierTransform():
    """
    implement Symmetric Fourier transform based on OGATA paper "Integration Based On Bessel Function", with a change of variable allowing to
    approximate the Symmetric Fourier transform, needed to approximate the structure factor of a set of data, by first approximating the pair
    correlation function (of just having the exact function), and taking the Fourier transform of the total pair correlation function .
    self....
    """

    def __init__(self, N, d=2, h=0.1):
        """
        Args:
            d (int): dimension of the space. Defaults to 2.
            N (int): number of sample points used to approximate the integral by a sum.
            h (float): step size in the sum. Defaults to 0.1.
            à ajouter les methods
        """
        if not isinstance(N, int):
            raise TypeError("N should be an integer.")
        self._h = h
        self.d = d
        self.N = N
        self._zeros = roots(d, N)  # Xi
        self.x = get_x(h, self._zeros)  # pi*psi(h*ksi/pi)/h
        self.kernel = jv(d/2 - 1, self.x)  # J_(d/2-1)(pi*psi(h*ksi))
        self.w = weight(d, self._zeros)  # (Y_0(pi*zeros)/J_1(pi*zeros))
        self.dpsi = d_psi(h * self._zeros)  # dpsi(h*ksi)
        self._factor = None

    def _f(self, r_vector, data_g):
        """given evaluations of the pair correlation function (g), it returns an interpolation of the total correlation function (h=g-1)

        Args:
            r_vector (np.array): vector containing the radius on which the pair correlation function is evaluated.
            data_g (np.array_like(r_vector)): vector containing the evaluations of the pair correlation function on r_vec.
        """
        data_f = data_g - 1
        self.f = interpolate.interp1d(
            r_vector, data_f, axis=0, fill_value='extrapolate', kind='cubic')
        return(self.f)

    def _k(self, k):
        return np.array(np.array(k))

    @property
    def _series_fac(self):
        if self._factor is None:
            self._factor = np.pi * self.w * self.kernel * \
                self.dpsi  # pi*w*J_(d/2-1)(x)*dpsi(h*zeros)
        return self._factor

    def _get_series(self, f, k, alpha):
        with np.errstate(divide="ignore"):  # numpy safely divides by 0
            args = np.divide.outer(self.x, k).T  # x/k
        # pi*w*J_(d/2-1)(x)*dpsi(h*zeros)f(x/k)J_(d/2-1)(x)*x**(d/2)
        return self._series_fac * (f(args) - 1*alpha) * (self.x**(self.d/2))

    def transform(self, k, g=None,  r_vector=None, data_g=None,):
        """Return an approximation of the symmetric Fourier transform of the total correlation function (h = g-1), and an estimation of the minimum confidence wave length.

        Args:
            k (np.array): vector containing the wavelength on which we want to approximate the structure factor.
            g (func): Pair correlation function if it's  known, else it will be approximated using data_g and r_vector. Defaults to None ( in this case r_vector and data_g should be provided).
            r_vector (np.array): vector containing the radius on which the pair correlation function is evaluated . Defaults to None.
            data_g (np.array_like(r_vector)): vector containing the evaluations of the pair correlation function on r_vec. Defaults to None.


        Returns:
            ret (np.array_like(k)): estimation of the fourier transform of the total correlation function.
            k_min (float): minimum confidence value of wavelength.
        """
        k = self._k(k)
        if g == None:
            f = self._f(r_vector, data_g)
            self.k_min = (np.pi * 3.2)/(self._h * np.max(r_vector))
            summation = self._get_series(f, k, alpha=0)  # pi*w*J0(x)

        else:
            self.k_min = np.min(k)
            summation = self._get_series(g, k, alpha=1)  # pi*w*J0(x)

        ret = np.empty(k.shape, dtype=summation.dtype)
        pi_factor = (2*np.pi)**(self.d/2)
        # 2pi/k**2*sum(pi*w*f(x/k)J_0(x)*dpsi(h*ksi)*x)
        ret = np.array(pi_factor * np.sum(summation,
                                          axis=-1) / np.array(k ** self.d))
        return (ret, self.k_min)


class StructureFactor(SymmetricFourierTransform):
    """implement various estimators of the structure factor of a point process.

    Args:
        SymmetricFourierTransform (class)
    """

    def __init__(self, data, intensity: float):
        """
        Args:
            data (np.array): data set of shape nxd array where, n is the number of data points and d is the dimension of the space (for now d=2).
            intensity (float): average number of data per unit volume.

        Raises:
            TypeError: [description]
            IndexError: [description]
            ValueError: [description]
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data should be an (nxd) NumPy array")

        try:
            self.n_data = data.shape[0]
            self.d = data.shape[1]
        except:
            raise IndexError("Input data should be an (nxd) NumPy array")

        if self.d != 2:
            raise ValueError("The library only supports 2D patterns so far")

        self.data = np.array(data)
        self.intensity = intensity
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]

    def get_scattering_intensity_estimate(self, L, max_k, n_k=None, arg="1D"):
        """compute the ensemble estimator described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5)
        which converges to the structure factor as n_data and the volume of the space goes to infinity.
        Notes:  The data should be simulated in a square.
                The allowed valued of wave vectors are 2*pi/n_data*k with k in Z*d

        Args:
            L (int): length of the square that contains the data.
            max_k (int): maximum k
            n_k (int): if arg=2D then n_k is the number of wave vector in each row. Defaults to None.
            arg (str): (1D or 2D), chose of evaluation of the structure factor on vector(1D), or meshgrid(2D). Defaults to "1D".

        Returns:
            norm_k (np.ndarray): wavelength of the wave vectors (x_k, y_k)
            si (np.ndarray_like(norm_k)): scattering intensity of the data on the wave vectors (x_k, y_k)

        """
        x_max = np.floor(max_k*L/(2*np.pi*np.sqrt(2)))

        if arg == "2D":
            x_grid = np.linspace(0, x_max, n_k)
            x_k, y_k = np.meshgrid(x_grid, x_grid)
        else:
            x_k = np.linspace(1, x_max, x_max)
            y_k = x_k

        self.x_k = x_k
        self.y_k = y_k
        self.norm_k = np.sqrt(x_k**2 + y_k**2)
        si_ = 0  # initial value of the sum in the scattering intensity
        x_data = self.x_data
        y_data = self.y_data
        n_data = self.n_data
        for i in range(0, n_data):
            # the sum in the formula of the scattering intensity
            si_ = si_ + np.exp(- 1j * (x_k * x_data[i] + y_k * y_data[i]))
        self.si = (1 / n_data)*np.abs(si_) ** 2
        return self.norm_k, self.si

    def plot_scattering_intensity_estimate(self, arg):
        """2D and  1D plot of the scattering intensity

        Args:
            arg (str): ("all","color_level" or "plot), is the plot visualization type.
        """
        x_k = self.x_k
        y_k = self.y_k
        si = self.si
        norm_k = np.sqrt(np.abs(x_k)**2 + np.abs(y_k)**2)
        ones_ = np.ones_like(x_k).T
        x_ones_ = np.linspace(
            np.min(norm_k), np.max(norm_k), np.max(x_k.shape))
        if arg == "all":
            if np.min(x_k.shape) == 1 or np.min(y_k.shape) == 1:
                raise ValueError(
                    "X_k, Y_k should be meshgrids or choose arg = 'plot'. ")
            else:
                fig, ax = plt.subplots(1, 3, figsize=(24, 7))
                ax[0].plot(self.x_data, self.y_data, 'b.')
                ax[0].title.set_text("data")
                ax[1].loglog(norm_k, si, 'k,')
                ax[1].loglog(x_ones_, ones_, 'r--')
                ax[1].legend(["Scattering intensity", "y=1"],
                             shadow=True, loc=1)
                ax[1].set_xlabel("wave length (k)")
                ax[1].set_ylabel("scattering intensity (SI(k))")
                ax[1].title.set_text("loglog plot")
                f_0 = ax[2].imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(
                    si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2.], cmap="PRGn")
                fig.colorbar(f_0, ax=ax[2])
                ax[2].title.set_text("scattering intensity")
                plt.show()
        elif arg == "plot":
            plt.loglog(norm_k, si, 'k,')
            plt.loglog(x_ones_, ones_, 'r--')
            plt.legend(['Scattering intensity', 'y=1'], loc=1)
            plt.xlabel("wave length (k)")
            plt.ylabel("Scattering intensity (SI(k))")
            plt.title("loglog plot")
            plt.show()
        elif arg == "color_level":
            if np.min(x_k.shape) == 1 or np.min(y_k.shape) == 1:

                raise ValueError(
                    "X_k, Y_k should be meshgrids or choose arg = 'plot'. ")
            else:
                f_0 = plt.imshow(np.log10(si), extent=[-np.log10(si).shape[1]/2., np.log10(
                    si).shape[1]/2., -np.log10(si).shape[0]/2., np.log10(si).shape[0]/2.], cmap="PRGn")
                plt.colorbar(f_0)
                plt.title("Scattering intensity")
                plt.show()
        else:
            raise ValueError(
                "arg should be one of the following str: 'all', 'plot' and 'color_level'.  ")

    def get_pcf_estimate(self, radius, args, correction_=None, r_vec=None, r_max=None, spar_=None):
        """compute the pair correlation function of data using the R package spatstat pcf.ppp, and pcf. fv.
         for more details see : "https://rdrr.io/cran/spatstat/man/pcf.ppp.html")

        Args:
            radius (float): radius of the ball containing the data on which the pair correlation function will be evaluated.
            args (str): If 'fv' then pcf.fv is used, if 'ppp' then pcf.ppp is used.
            correction_ (str): if args='ppp' : correction_  should be one of: "translate", "Ripley", "isotropic", "best", "good" , "all". If args='fv' : correction should be one of: "a", "b", "c" or "d" .  Defaults to None.
            r_vec (np.array): if arg='fv' : r_vec is the vector of radius on which g will be evaluated. it's preferred to keep the default or to set an r_max. If args='ppp' : keep the Default. Defaults to None.
            r_max (float): if args='fv' : r_max is the maximum radius on which g will be evaluated. if args='ppp' : keep the default.Defaults to None.
            spar_ (float): sparsity parameter. Defaults to None.

        Returns:
            pcf_estimation_pd (pd_dataframe): pandas data frame containing a column of radius, a column of ones representing the theoretical values of the pcf (corresponding to a Poisson point process), the remaining columns contains the pcf approximations.
        """
        if args not in ["fv", "ppp"]:
            raise ValueError(" args should be 'fv', or 'ppp'")
        if not np.isscalar(radius):
            raise ValueError("radius must be a scalar")

        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        spatstat = rpackages.importr('spatstat')
        disc = robjects.r("disc")
        center = robjects.r('c')
        r_base = importr('base')
        ppp = robjects.r('ppp')
        pcf = robjects.r('pcf')
        Kest = robjects.r('Kest')

        x_data = self.x_data
        y_data = self.y_data
        x_data_r = robjects.vectors.FloatVector(x_data)
        y_data_r = robjects.vectors.FloatVector(y_data)
        data_r = ppp(x_data_r, y_data_r, window=disc(radius, center(0, 0)))

        if args == "ppp":
            if correction_ not in ["translate", "Ripley", "isotropic", "best", "good", "all", None]:
                raise ValueError(
                    "correction should be one of the following str: 'translate', 'Ripley', 'isotropic', 'best', 'good' , 'all', 'none'.")
            if spar_ is not None:
                pcf_estimation = pcf(data_r, spar=spar_,
                                     correction=correction_)
            else:
                pcf_estimation = pcf(data_r, correction=correction_)

        if args == "fv":
            if correction_ not in ["a", "b", "c", "d", None]:
                raise ValueError(
                    "correction_ should be one of the following str: 'a', 'b', 'c', 'd', None.")
            # Transfert ppp_data as variables in  R.
            robjects.globalenv['data_r'] = data_r
            if r_max != None:
                robjects.globalenv['r_max'] = r_max
                robjects.r('kest_data = Kest(data_r, rmax=r_max)')
            elif r_vec is not None:
                robjects.globalenv['r_vec'] = r_vec
                r_vec_r = robjects.vectors.FloatVector(r_vec)
                robjects.r('kest_data = Kest(data_r, r=r_vec)')
            else:
                robjects.r('kest_data = Kest(data_r)')
            robjects.globalenv['method_'] = correction_
            if spar_ is not None:
                robjects.globalenv['spar_'] = spar_
                pcf_estimation = robjects.conversion.rpy2py(robjects.r(
                    'pcf.fv(kest_data, spar=spar_, all.knots=FALSE,  keep.data=TRUE,  method=method_)'))
            else:
                pcf_estimation = robjects.conversion.rpy2py(robjects.r(
                    'pcf.fv(kest_data, all.knots=FALSE,  keep.data=TRUE,  method=method_)'))
        pcf_data_farme = pd.DataFrame.from_records(
            pcf_estimation)
        pcf_data_farme.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.pcf_estimation_pd = pd.DataFrame.from_records(
            pcf_data_farme).fillna(0)  # fill nan values with zeros

        return self.pcf_estimation_pd

    def plot_pcf_estimate(self, args):
        """ plot the pair correlation function estimation approximated by get_pcf_estimate.

        Args:
            args (str): ('pcf', 'trans', 'iso' or 'un'), type of pcf already approximated.
        """

        pcf_estimation_pd = self.pcf_estimation_pd
        g_key = (pcf_estimation_pd.keys()).tolist()
        if g_key.count(args) == 0:
            raise ValueError(
                "The data frame does not contains the chosen args. Check pcf_estimation_pd.keys(). ")
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

    def get_fourier_estimate(self, args=None, arg_2=None, g=None, N=None, h=0.1, k=None):
        """compute an approximation of the structure factor by evaluating the Symmetric Fourier transform of the approximated pair correlation function or the exact if it's known.

        Args:
            args (str):  ('pcf', 'ppp_trans', 'ppp_iso' or 'ppp_un'). Specifies the method chosen to approximate the pcf.
                 'pcf' if pcf.fv is used.
                 'trans', 'iso' or 'un': if pcf.ppp is used, and it specifies which edge correction is used. Defaults to None.
            arg_2 (str): [description]. Defaults to None.
            g (func):  the pair correlation function if it is known and we need to evaluate the structure factor directly. Else approximated by get_pcf_estimate. Defaults to None.
            N (int): see the class SymmetricFourierTransform. Defaults to None.
            h (float): see the class SymmetricFourierTransform. Defaults to 0.1.
            k (np.array): see the class SymmetricFourierTransform. Defaults to None.

        Returns:
             if arg_2 = estimation_1:
                norm_k (np.array): array containing the norms of the elements of the wave vector on which the structure facture is approximated.
                sf_estimation (np.array_like(norm_k)): array containing the estimation of the structure factor.
            if arg_2 = estimation_2:
                sf_estimation_2 (np.array): array containing th estimation of the structure factor.
                self.k_min (float): approximation of the reliable minimum wavelength.
        """

        intensity = self.intensity
        pcf_estimation_pd = self.pcf_estimation_pd
        g_key = (pcf_estimation_pd.keys()).tolist()
        intensity = self.intensity
        if g is not None and not callable(g):
            raise TypeError(
                "g should be of type function representing the pair correlation function.")

        if g == None and g_key.count(args) == 0:
            raise ValueError(
                "The data frame does not contains the chosen args. Check pcf_estimation_pd.keys() to plot one of them. ")

        g_to_plot = np.array(pcf_estimation_pd[args])
        r_vec = np.array(pcf_estimation_pd["r"])
        g_theo = np.array(pcf_estimation_pd["theo"])
        h_estimation = pcf_estimation_pd[args] - 1

        if arg_2 == "estimation_1":
            h_estimation[0] = -1
            transformer = HankelTransform(order=0, max_radius=max(
                pcf_estimation_pd['r']), n_points=pcf_estimation_pd['r'].shape[0])
            sf_estimation = 1 + intensity*transformer.qdht(h_estimation)
            norm_k = transformer.kr
            ones_ = np.ones_like(sf_estimation)

            fig, ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, 'r--', label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c='k', s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, 'b')
            ax[0].title.set_text("Pair correlation function ")
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[1].plot(norm_k[1:], sf_estimation[1:], 'b')
            ax[1].scatter(norm_k, sf_estimation, c='k', s=1, label="sf")
            ax[1].plot(norm_k, ones_, 'r--', label="y=1")
            ax[1].legend()
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('S(k)')
            ax[1].title.set_text('structure factor of data')
            plt.show()
            return (norm_k, sf_estimation)

        if arg_2 == "estimation_2":
            if N == None:
                N = 1000
            super().__init__(d=self.d, N=N, h=h)

            sf, self.k_min = super().transform(k=k, g=g, data_g=g_to_plot, r_vector=r_vec)
            print("The reliable minimum wavelength is :",  self.k_min)
            sf_estimation_2 = 1 + intensity * sf
            sf_interpolate = interpolate.interp1d(
                k, sf_estimation_2, axis=0, fill_value='extrapolate', kind='cubic')
            ones_ = np.ones(sf_estimation_2.shape).T
            fig, ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, 'r--', label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c='k', s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, 'b')
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[0].title.set_text("Pair correlation function ")
            ax[1].plot(k[1:], sf_estimation_2[1:], 'b')
            ax[1].scatter(k, sf_estimation_2, c='k', s=1, label="sf")
            ax[1].plot(self.k_min, sf_interpolate(
                self.k_min), "ro", label="reliable k_min")
            ax[1].plot(k, ones_, 'r--', label="y=1")
            ax[1].legend()
            ax[1].set_xlabel('k')
            ax[1].set_ylabel('S(k)')
            ax[1].title.set_text('structure factor of data')
            plt.show()
            return(sf_estimation_2, self.k_min)
