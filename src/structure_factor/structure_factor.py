from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from structure_factor.utils import (
    estimate_scattering_intensity,
    SymmetricFourierTransform,
)
import numpy as np
import matplotlib.pyplot as plt
from pyhank import HankelTransform
from scipy import interpolate
import pandas as pd
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
rpy2.robjects.numpy2ri.activate()
pandas2ri.activate()


# todo give explicit names to "args, arg1, arg2" arguments


class StructureFactor:
    """implement various estimators of the structure factor of a point process.

    Args:
        SymmetricFourierTransform (class)
    """

    def __init__(self, data: np.ndarray, intensity: float):
        """
        Args:
            data (np.array): data set of shape nxd array where, n is the number of data points and d is the dimension of the space (for now d=2).
            intensity (float): average number of data per unit volume.

        Raises:
            TypeError: [description]
            IndexError: [description]
            ValueError: [description]
        """
        # todo mettre tous les atttribus ici comme k_min
        # todo enlever la dependence en x_data, y_data et garder que data
        if data.ndim != 2 and data.shape[1] != 2:
            raise ValueError("data must be nx2 array")
        self.n_data, self.d = data.shape
        self.data = data
        self.intensity = intensity
        self.x_data = data[:, 0]
        self.y_data = data[:, 1]

        self.norm_wave_vector = None
        self.scattering_intensity = None

    def estimate_scattering_intensity(self, L, maximum_wave, meshgrid_size=None):
        # todo replace the link below to the link of our future paper :).
        """compute the ensemble estimator described in http://www.scoste.fr/survey_hyperuniformity.pdf.(equation 4.5).
        This estimation converges to the structure factor in the thermodynamic limits.

        Notes:  The data should be simulated inside a cube.
                The allowed values of wave vectors are the points of the dual of the lattice having fundamental cell the cubic window .
                This is represented inside wave_vectors which we defines as:
                wave_vectors = 2*pi*k_vector/L
                where, k_vector is a vector of integer from 1 into maximum_k, and L in the length side of the cubic window.

        Args:
            L (int): length of the square that contains the data.
            maximum_wave (int): maximum norm of the wave vector
            meshgrid_size (int): if the requested evaluation is on a meshgrid,  then meshgrid_size is the number of wave vector in each row of the meshgrid. Defaults to None.


        Returns:
            norm_wave_vector (np.ndarray): wavelength of the wave vectors.
            scattering_intensity (np.ndarray_like(norm_wave_vector)): scattering intensity of the data evaluated on the wave vectors.

        """

        maximum_k = np.floor(maximum_wave * L / (2 * np.pi))
        if meshgrid_size is None:
            x = 2 * np.pi / L * np.linspace(1, maximum_k, int(maximum_k))
            wave_vector = np.column_stack((x, x))
        else:
            x_grid = np.linspace(0, maximum_wave, int(meshgrid_size))
            X, Y = np.meshgrid(x_grid, x_grid)
            wave_vector = np.column_stack((X.ravel(), Y.ravel()))

        norm_wave_vector = np.linalg.norm(wave_vector, axis=1)
        scattering_intensity = estimate_scattering_intensity(wave_vector, self.data)
        if meshgrid_size is not None:
            norm_wave_vector = norm_wave_vector.reshape(X.shape)
            scattering_intensity = scattering_intensity.reshape(X.shape)

        self.norm_wave_vector = norm_wave_vector
        self.scattering_intensity = scattering_intensity

        return norm_wave_vector, scattering_intensity

    def plot_scattering_intensity_estimate(self, arg):
        """2D and  1D plot of the scattering intensity

        Args:
            arg (str): ("all","color_level" or "plot), is the type of the requested plot.
        """
        norm_wave_vector = self.norm_wave_vector
        scattering_intensity = self.scattering_intensity
        if arg == "all":
            if np.min(norm_wave_vector.shape) == 1:
                raise ValueError(
                    "the scattering intensity should be evaluated on a meshgrid or choose arg = 'plot'. "
                )
            else:
                fig, ax = plt.subplots(1, 3, figsize=(24, 7))
                ax[0].plot(self.x_data, self.y_data, "b.")
                ax[0].title.set_text("data")
                ax[1].loglog(norm_wave_vector, scattering_intensity, "k,")
                ax[1].loglog(norm_wave_vector, np.ones_like(norm_wave_vector), "r--")
                ax[1].legend(["Scattering intensity", "y=1"], shadow=True, loc=1)
                ax[1].set_xlabel("norm wave vector")
                ax[1].set_ylabel("scattering intensity")
                ax[1].title.set_text("loglog plot")
                log_scattering_intensity = np.log10(scattering_intensity)
                m, n = log_scattering_intensity.shape
                m /= 2
                n /= 2

                f_0 = ax[2].imshow(
                    log_scattering_intensity,
                    extent=[-n, n, -m, m],
                    cmap="PRGn",
                )
                fig.colorbar(f_0, ax=ax[2])
                ax[2].title.set_text("scattering intensity")
                plt.show()
        elif arg == "plot":
            plt.loglog(norm_wave_vector, scattering_intensity, "k,")
            plt.loglog(norm_wave_vector, np.ones_like(norm_wave_vector), "r--")
            plt.legend(["Scattering intensity", "y=1"], loc=1)
            plt.xlabel("wave length (k)")
            plt.ylabel("Scattering intensity (SI(k))")
            plt.title("loglog plot")
            plt.show()
        elif arg == "color_level":
            if np.min(norm_wave_vector.shape) == 1:
                raise ValueError(
                    "the scattering intensity should be evaluated on a meshgrid or choose arg = 'plot'. "
                )
            else:
                # todo changer les log10 comme en haut ligne 220
                f_0 = plt.imshow(
                    np.log10(scattering_intensity),
                    extent=[
                        -np.log10(scattering_intensity).shape[1] / 2.0,
                        np.log10(scattering_intensity).shape[1] / 2.0,
                        -np.log10(scattering_intensity).shape[0] / 2.0,
                        np.log10(scattering_intensity).shape[0] / 2.0,
                    ],
                    cmap="PRGn",
                )
                plt.colorbar(f_0)
                plt.title("Scattering intensity")
                plt.show()
        else:
            raise ValueError(
                "arg should be one of the following str: 'all', 'plot' and 'color_level'.  "
            )

    def estimate_pcf(
        self, radius, args, correction_=None, r_vec=None, r_max=None, spar_=None
    ):
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

        utils = rpackages.importr("utils")
        utils.chooseCRANmirror(ind=1)
        spatstat = rpackages.importr("spatstat")
        disc = robjects.r("disc")
        center = robjects.r("c")
        r_base = importr("base")
        ppp = robjects.r("ppp")
        pcf = robjects.r("pcf")
        Kest = robjects.r("Kest")

        x_data = self.x_data
        y_data = self.y_data
        x_data_r = robjects.vectors.FloatVector(x_data)
        y_data_r = robjects.vectors.FloatVector(y_data)
        data_r = ppp(x_data_r, y_data_r, window=disc(radius, center(0, 0)))

        if args == "ppp":
            if correction_ not in [
                "translate",
                "Ripley",
                "isotropic",
                "best",
                "good",
                "all",
                None,
            ]:
                raise ValueError(
                    "correction should be one of the following str: 'translate', 'Ripley', 'isotropic', 'best', 'good' , 'all', 'none'."
                )
            if spar_ is not None:
                pcf_estimation = pcf(data_r, spar=spar_, correction=correction_)
            else:
                pcf_estimation = pcf(data_r, correction=correction_)

        if args == "fv":
            if correction_ not in ["a", "b", "c", "d", None]:
                raise ValueError(
                    "correction_ should be one of the following str: 'a', 'b', 'c', 'd', None."
                )
            # Transfert ppp_data as variables in  R.
            robjects.globalenv["data_r"] = data_r
            if r_max != None:
                robjects.globalenv["r_max"] = r_max
                robjects.r("kest_data = Kest(data_r, rmax=r_max)")
            elif r_vec is not None:
                robjects.globalenv["r_vec"] = r_vec
                r_vec_r = robjects.vectors.FloatVector(r_vec)
                robjects.r("kest_data = Kest(data_r, r=r_vec)")
            else:
                robjects.r("kest_data = Kest(data_r)")
            robjects.globalenv["method_"] = correction_
            if spar_ is not None:
                robjects.globalenv["spar_"] = spar_
                pcf_estimation = robjects.conversion.rpy2py(
                    robjects.r(
                        "pcf.fv(kest_data, spar=spar_, all.knots=FALSE,  keep.data=TRUE,  method=method_)"
                    )
                )
            else:
                pcf_estimation = robjects.conversion.rpy2py(
                    robjects.r(
                        "pcf.fv(kest_data, all.knots=FALSE,  keep.data=TRUE,  method=method_)"
                    )
                )
        pcf_data_farme = pd.DataFrame.from_records(pcf_estimation)
        pcf_data_farme.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.pcf_estimation_pd = pd.DataFrame.from_records(pcf_data_farme).fillna(
            0
        )  # fill nan values with zeros

        return self.pcf_estimation_pd

    def plot_pcf_estimate(self, args):
        """plot the pair correlation function estimation approximated by estimate_pcf.

        Args:
            args (str): ('pcf', 'trans', 'iso' or 'un'), type of pcf already approximated.
        """

        pcf_estimation_pd = self.pcf_estimation_pd
        g_key = (pcf_estimation_pd.keys()).tolist()
        if g_key.count(args) == 0:
            raise ValueError(
                "The data frame does not contains the chosen args. Check pcf_estimation_pd.keys(). "
            )
        r_vec = np.array(pcf_estimation_pd["r"])
        g_to_plot = np.array(pcf_estimation_pd[args])
        g_theo = np.array(pcf_estimation_pd["theo"])
        plt.plot(r_vec, g_theo, "r--", label="y=1")
        plt.scatter(r_vec, g_to_plot, c="k", s=1, label="pcf")
        plt.plot(r_vec, g_to_plot, "b")
        plt.legend()
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Pair correlation function ")

    # todo add a plot_fourier function should not plot anything, add a plot_estimate_fourier and delete all plots from this function
    def estimate_fourier(self, args=None, arg_2=None, g=None, N=None, h=0.1, k=None):
        """compute an approximation of the structure factor by evaluating the Symmetric Fourier transform of the approximated pair correlation function or the exact if it's known.

        Args:
            args (str):  ('pcf', 'ppp_trans', 'ppp_iso' or 'ppp_un'). Specifies the method chosen to approximate the pcf.
                 'pcf' if pcf.fv is used.
                 'trans', 'iso' or 'un': if pcf.ppp is used, and it specifies which edge correction is used. Defaults to None.
            arg_2 (str): [description]. Defaults to None.
            g (func):  the pair correlation function if it is known and we need to evaluate the structure factor directly. Else approximated by estimate_pcf. Defaults to None.
            N (int): see the class SymmetricFourierTransform. Defaults to None.
            h (float): see the class SymmetricFourierTransform. Defaults to 0.1.
            k (np.array): see the class SymmetricFourierTransform. Defaults to None.

        Returns:
             if arg_2 = estimation_1:
                norm_wave_vector (np.array): array containing the norms of the elements of the wave vector on which the structure facture is approximated.
                sf_estimation (np.array_like(norm_wave_vector)): array containing the estimation of the structure factor.
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
                "g should be of type function representing the pair correlation function."
            )

        if g is None and g_key.count(args) == 0:
            raise ValueError(
                "The data frame does not contains the chosen args. Check pcf_estimation_pd.keys() to plot one of them. "
            )

        g_to_plot = np.array(pcf_estimation_pd[args])
        r_vec = np.array(pcf_estimation_pd["r"])
        g_theo = np.array(pcf_estimation_pd["theo"])
        h_estimation = pcf_estimation_pd[args] - 1

        if arg_2 == "estimation_1":
            h_estimation[0] = -1
            transformer = HankelTransform(
                order=0,
                max_radius=max(pcf_estimation_pd["r"]),
                n_points=pcf_estimation_pd["r"].shape[0],
            )
            sf_estimation = 1 + intensity * transformer.qdht(h_estimation)
            norm_wave_vector = transformer.kr
            ones_ = np.ones_like(sf_estimation)

            fig, ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, "r--", label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c="k", s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, "b")
            ax[0].title.set_text("Pair correlation function ")
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[1].plot(norm_wave_vector[1:], sf_estimation[1:], "b")
            ax[1].scatter(norm_wave_vector, sf_estimation, c="k", s=1, label="sf")
            ax[1].plot(norm_wave_vector, ones_, "r--", label="y=1")
            ax[1].legend()
            ax[1].set_xlabel("k")
            ax[1].set_ylabel("S(k)")
            ax[1].title.set_text("structure factor of data")
            plt.show()
            return norm_wave_vector, sf_estimation

        if arg_2 == "estimation_2":
            if N is None:
                N = 1000

            # todo mettre les noms par order

            transformer = SymmetricFourierTransform(d=self.d, N=N, h=h)
            # todo il faut sortir de transform kmin car il n'est plus un atribu
            sf, self.k_min = transformer.transform(
                k=k, g=g, data_g=g_to_plot, r_vector=r_vec
            )
            print("The reliable minimum wavelength is :", self.k_min)
            sf_estimation_2 = 1 + intensity * sf
            sf_interpolate = interpolate.interp1d(
                k, sf_estimation_2, axis=0, fill_value="extrapolate", kind="cubic"
            )
            ones_ = np.ones(sf_estimation_2.shape).T
            fig, ax = plt.subplots(1, 2, figsize=(24, 7))
            ax[0].plot(r_vec, g_theo, "r--", label="y=1")
            ax[0].scatter(r_vec, g_to_plot, c="k", s=1, label="pcf")
            ax[0].plot(r_vec, g_to_plot, "b")
            ax[0].legend()
            ax[0].set_xlabel("r")
            ax[0].set_ylabel("g(r)")
            ax[0].title.set_text("Pair correlation function ")
            ax[1].plot(k[1:], sf_estimation_2[1:], "b")
            ax[1].scatter(k, sf_estimation_2, c="k", s=1, label="sf")
            ax[1].plot(
                self.k_min, sf_interpolate(self.k_min), "ro", label="reliable k_min"
            )
            ax[1].plot(k, ones_, "r--", label="y=1")
            ax[1].legend()
            ax[1].set_xlabel("k")
            ax[1].set_ylabel("S(k)")
            ax[1].title.set_text("structure factor of data")
            plt.show()
            return sf_estimation_2, self.k_min
