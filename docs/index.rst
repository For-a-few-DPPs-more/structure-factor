.. structure-factor documentation master file, created by
   sphinx-quickstart on Wed Jul 21 15:25:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to structure-factor's documentation:
============================================

This is the documentation of the ``structure-factor`` project: an open source Python toolbox designed to approximate the structure factor of a `stationary point process <https://en.wikipedia.org/wiki/Point_process#Stationarity>`_, test its effective hyperuniformity and identify its class of hyperuniformity.

- source code on `GitHub <https://github.com/For-a-few-DPPs-more/structure-factor>`_
- tutorial `Jupyter notebook <https://github.com/For-a-few-DPPs-more/structure-factor/tree/main/notebooks>`_

Introduction
------------

In condensed matter physics, it has been observed for some particle systems that, the variance of the number of points in a large window is lower than expected, a phenomenon called hyperuniformity.
To study the hyperuniformity of a given point process, common practice in statistical physics is to estimate a spectral measure called the `structure factor <https://en.wikipedia.org/wiki/Structure_factor>`_, the behavior of which around zero is a indicator of hyperuniformity.
The structure factor of a point process is defined via the Fourier transform of its total correlation function; see :cite:`Cos21` for more details.

Installation
------------

See the `"Installation" section in the README file on GitHub <https://github.com/For-a-few-DPPs-more/structure-factor#installation>`_.

Getting started
---------------

Please refer to

- the tutorial `Jupyter notebook <https://github.com/For-a-few-DPPs-more/structure-factor/tree/main/notebooks>`_, and
- the different sections of the present documentation page.

Contents
--------

.. toctree::
   :maxdepth: 2

   hyperuniformity
   point_pattern
   spatial_windows
   structure_factor
   transforms
   utils
   bibliography/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
