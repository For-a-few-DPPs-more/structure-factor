import rpy2.robjects.packages as rpackages


def install_r_package(*names, update=True):
    utils = rpackages.importr("utils")
    for name in names:
        if rpackages.isinstalled(name) and not update:
            continue
        # Choose mirror (internet access required) if update = True
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(name)


class SpatstatInterface:
    """See also https://github.com/spatstat/spatstat"""

    SUBPACKAGES = ("core", "data", "geom", "linnet", "sparse", "spatstat", "utils")
    EXTENSIONS = ("gui", "Knet", "local", "sphere")

    def __init__(self, update=True):
        """Construct an interface with ``spatstat`` R package `github.com/spatstat <https://github.com/spatstat/spatstat>`_.
        If ``spatstat`` is already installed is it updated according to ``update`` otherwise it is installed.

        :param update: Install latest version of ``spatstat`` (internet access required), defaults to True
        :type update: bool, optional

        .. seealso::

            :py:meth:`SpatstatInterface.import_package`
        """
        install_r_package("spatstat", update=update)
        for pkg in self.SUBPACKAGES + self.EXTENSIONS:
            setattr(self, pkg, None)

    def import_package(self, *names, update=True):
        """Import spatstat subpackages or extensions given by ``names``, made accessible via the corresponding ``name`` attribute.
        If the package is already present is it updated according to ``update`` otherwise it is installed.

        :param update: Install latest version of the corresponding package (internet access required), defaults to True
        :type update: bool, optional

        .. note::

            As mentioned on `github.com/spatstat <https://github.com/spatstat/spatstat>`_, when ``spatstat`` is installed all subpackages will automatically be installed.
            On the contrary ``spatstat`` extensions must be installed separately.

        .. seealso::

            The list of ``spatstat`` subpackages and extensions is available at `github.com/spatstat <https://github.com/spatstat/spatstat>`_
        """
        self.check_package_name(*names)
        spatstat = "spatstat"
        for name in names:
            pkg = spatstat if name == spatstat else f"{spatstat}.{name}"
            install_r_package(pkg, update=update)
            setattr(self, name, rpackages.importr(pkg))

    def check_package_name(self, *names):
        """Check whether ``names`` are valid ``spatstat`` subpackages or extension.

        :raises ValueError: if some names are invalid.

        .. seealso::

            The list of subpackages and extensions is available at `github.com/spatstat <https://github.com/spatstat/spatstat>`_
        """
        wrong_names = set(names).difference(self.SUBPACKAGES + self.EXTENSIONS)
        if wrong_names:
            raise ValueError(
                f"{wrong_names} are invalid spatstat subpackages {self.SUBPACKAGES} or extensions {self.EXTENSIONS}."
            )
