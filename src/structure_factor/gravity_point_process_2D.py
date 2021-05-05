import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


class Gravity_point_process_2D():

    def get_poisson_point_process(self, arg, intensity, r=None, x_center=None, y_center=None, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        arg: (str) 'rect' or 'disc'
             if 'rect' then the Poisson point process is simulated in a rectangle
             if 'disc' then the Poisson point process is simulated in a disc
        r: (arg ='disc'), r is he radius of the disc, which is the support of Poisson point process
        x_center: x coordinate of the center
        y_center: y coordinate of the center
        x_min, x_max:(arg='rect') minimum and maximum x coordinates of the rectangle containing the Poisson point process
        y_min, y_max:(arg='rect') minimum and maximum y coordinates of the rectangle containing the Poisson point process
        """

        if arg not in ["rect", "disc"]:
            raise ValueError(
                "arg should be one of the following str: 'rect', 'disc'.")
        self.intensity_pois = intensity

        if arg == "disc":
            self.r = r
            total_area = np.pi * r ** 2  # area of disk
            # Poisson number of points
            n_pois = np.random.poisson(intensity * total_area)
            # angular coordinates
            theta = 2 * np.pi * np.random.uniform(0, 1, n_pois)
            rho = r * np.sqrt(np.random.uniform(0, 1, n_pois)
                              )  # radial coordinates
            # x coordinate of the poisson in the disc centered at the origine
            x_pois_ = rho * np.cos(theta)
            # y coordinate of the poisson in the disc centered at the origine
            y_pois_ = rho * np.sin(theta)
            # x coordinate of the poisson in the disc centered at (x_center,0)
            x_pois = x_pois_ + x_center
            # y coordinate of the poisson in the disc centered at (0, y_center)
            y_pois = y_pois_ + y_center

        if arg == "rect":
            x_rect = x_max - x_min  # rectangle dimension
            y_rect = y_max - y_min  # rectangle dimension
            total_area = x_rect * y_rect  # total area of the rectangle
            # Poisson number of points
            n_pois = scipy.stats.poisson(intensity * total_area).rvs()
            x_pois = (x_rect*scipy.stats.uniform.rvs(0, 1, ((n_pois, 1))) +
                      x_min).reshape(-1,)  # x coordinates of Poisson points
            y_pois = (y_rect*scipy.stats.uniform.rvs(0, 1, ((n_pois, 1))) +
                      y_min).reshape(-1,)  # y coordinates of Poisson points

        pois_data_c = x_pois + 1j * y_pois  # poisson data in the complex plane
        self.n_pois = n_pois  # number of point of the poisson point process
        # the constante c related to the volume of the basin
        self.c = (n_pois - 1) * np.pi / total_area
        ordered_index = (np.abs(pois_data_c)).argsort(axis=0)  # index
        # .reshape((n_pois, 1)) # rearrangement of poisson by increasing distance
        self.pois_data_c = (pois_data_c[ordered_index])
        # poisson point process data
        pois_data_vec = np.array([x_pois, y_pois])
        return (pois_data_vec)

    def _force(self, pois_data_c_minus_m, pois_data_c_m):
        c = self.c
        # denominator of the fraction in the sum of the force function
        deno = np.abs(pois_data_c_m - pois_data_c_minus_m) ** 2
        # x coordinate of numerator of the fraction in the sum of the force function
        x_num = np.real(pois_data_c_m) - np.real(pois_data_c_minus_m)
        # y coordinate of numerator of the fraction in the sum of the force function
        y_num = np.imag(pois_data_c_m) - np.imag(pois_data_c_minus_m)
        # x coordinate of the force function on pois_data_c_m
        x_F = np.sum(x_num / deno) - c * np.real(pois_data_c_m)
        # y coordinate of the force function on pois_data_c_m
        y_F = np.sum(y_num / deno) - c * np.imag(pois_data_c_m)
        return (x_F, y_F)

    def get_push_point_process(self, t_max):
        pois_data_c = self.pois_data_c
        n_pois = self.n_pois
        c = self.c
        epsilon = np.pi / (40 * c)  # epsilon
        # initial push data complex vector
        push_data_c = np.zeros((pois_data_c.shape), dtype="complex")
        for m in range(0, n_pois):
            pois_data_c_minus_m = (np.delete(pois_data_c, m)).reshape(
                n_pois - 1, 1)  # pois_data_c without pois_data_c[m]
            pois_data_c_m = pois_data_c[m]
            for t in range(0, t_max):
                (x_F, y_F) = self._force(pois_data_c_minus_m, pois_data_c_m)
                # updated coordinate of pois_data_c_m at time t
                pois_data_c_m = pois_data_c_m + epsilon*(x_F + 1j*y_F)
            push_data_c[m] = pois_data_c_m
        self.push_data_c = push_data_c
        push_data_vec = np.array([np.real(push_data_c), np.imag(push_data_c)])
        return(push_data_vec)

    def get_garvity_equilibrium_point_process(self, t_max):
        pois_data_c = self.pois_data_c  # equi data at t=0
        equi_data_c = np.copy(pois_data_c)
        n_pois = self.n_pois
        c = self.c
        epsilon = 0.1
        for t in range(0, t_max):
            for m in range(0, n_pois):
                equi_data_c_minus_m = (np.delete(equi_data_c, m)).reshape(
                    n_pois - 1, 1)  # pois_data_c without pois_data_c[m]
                equi_data_c_m = equi_data_c[m]
                (x_F, y_F) = self._force(equi_data_c_minus_m, equi_data_c_m)
                equi_data_c_m = equi_data_c_m + epsilon*(x_F + 1j * y_F)
                equi_data_c[m] = equi_data_c_m
        self.equi_data_c = equi_data_c
        equi_data_vec = np.array([np.real(equi_data_c), np.imag(equi_data_c)])
        return(equi_data_vec)

    def plot_point_process(self, arg):
        """
        arg: (str) an element of the list ['poisson', 'push', 'gravity_equilibrium', 'all']
        """
        if arg not in ["poisson", "push", "gravity_equilibrium", "all"]:
            raise ValueError(
                "arg should be one of the following str: 'poisson', 'push', 'gravity_equilibrium'.")
        if arg == "poisson":
            plt.figure(figsure(5, 5))
            plt.scatter(np.real(self.pois_data_c),
                        np.imag(self.pois_data_c), s=1, c='k')
            plt.title("Poison point process")
        if arg == "push":
            plt.figure(figsure(5, 5))
            plt.scatter(np.real(self.push_data_c),
                        np.imag(self.push_data_c), s=1, c='k')
            plt.title("Push point process")
        if arg == "gravity_equilibrium":
            plt.figure(figsure(5, 5))
            plt.scatter(np.real(self.equi_data_c),
                        np.imag(self.equi_data_c), s=1, c='k')
            plt.title("Gravity equilibrium point process")
        if arg == "all":
            fig, ax = plt.subplots(1, 3, figsize=(24, 7))
            ax[0].scatter(np.real(self.pois_data_c),
                          np.imag(self.pois_data_c), s=1, c='k')
            ax[0].title.set_text("Poisson point process")
            ax[1].scatter(np.real(self.push_data_c),
                          np.imag(self.push_data_c), s=1, c='k')
            ax[1].title.set_text('Push point process')
            ax[2].scatter(np.real(self.equi_data_c),
                          np.imag(self.equi_data_c), s=1, c='k')
            ax[2].title.set_text("Gravity equilibrium point process")
