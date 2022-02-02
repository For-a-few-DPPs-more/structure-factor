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

Hyperuniformity is the study of stationary point processes with a sub-Poisson variance of the number of points in a large window. For the homogeneous Poisson point process, the variance of the number of points that fall in a large window is of the order of the window volume. In contrast, for hyperuniform (HU) point processes, the corresponding variance is much lower than the volume of that window, with a ratio going to zero :cite:`Tor18`.

In the context of amorphous structures, hyperuniformity
implies a hidden form of order, in which the system remains
macroscopically uniform, despite not being crystalline. The
concept of hyperuniformity sheds light on a variety of seemingly
unrelated fields, including density fluctuations in the early universe, biological tissue, statistical physics, colloidal
or granular packings, micro fluids, driven nonequilibrium
systems... :cite:`KlaAl19`.

There are many candidate HU processes in the physics literature, but rigorously proving that a point process is HU is usually difficult. It is thus desirable to have standardized numerical tests of hyperuniformity. A common practice in statistical physics is to estimate a spectral measure called the structure factor, the behavior of which around zero is a sign of hyperuniformity :cite:`Cos21`.

This Python toolbox gathers many estimators of the structure factor, along with a numerical test of effective hyperuniformity and the corresponding possible class of hyperuniformity.

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
   :maxdepth: 3

   hyperuniformity
   point_pattern
   spatial_windows
   structure_factor
   spectral_estimators
   tapers
   transforms
   utils
   bibliography/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
