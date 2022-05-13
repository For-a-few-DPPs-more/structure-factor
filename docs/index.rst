.. structure-factor documentation master file, created by
   sphinx-quickstart on Wed Jul 21 15:25:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to structure-factor's documentation:
============================================

This is the documentation of the ``structure-factor`` open-source Python project which currently collects

- various estimators of the structure factor,
- several diagnostics of hyperuniformity,

for stationary and isotropic point processes.

- source code `GitHub <https://github.com/For-a-few-DPPs-more/structure-factor>`_
- tutorial `Jupyter notebook <https://github.com/For-a-few-DPPs-more/structure-factor/tree/main/notebooks>`_

Introduction
------------

Hyperuniformity is the study of stationary point processes with a sub-Poisson variance of the number of points in a large window. For the homogeneous Poisson point process, the variance of the number of points that fall in a large window is of the order of the window volume. In contrast, for hyperuniform (HU) point processes, the corresponding variance is much lower than the volume of that window, with a ratio going to zero :cite:`Tor18`.

In the context of amorphous structures, hyperuniformity implies a hidden form of order, in which the system remains macroscopically uniform, despite not being crystalline.
The concept of hyperuniformity sheds light on a variety of seemingly unrelated fields, including density fluctuations in the early universe, biological tissue, statistical physics, colloidal or granular packings, micro fluids, driven nonequilibrium systems... :cite:`KlaAl19`.

There are many candidate HU processes in the physics literature, but rigorously proving that a point process is HU is usually difficult. It is thus desirable to have standardized numerical tests of hyperuniformity. A common practice in statistical physics is to estimate a spectral measure called the structure factor, the behavior of which around zero is a sign of hyperuniformity :cite:`Cos21`.

This Python toolbox gathers the estimators of the structure factor surveyed in :cite:`HGBLR:22`, along with a new statistical test of hyperuniformity asymptotically valid :cite:`HGBLR:22`, the test of effective hyperuniformity :cite:`Tor18` and the corresponding possible class of hyperuniformity :cite:`Cos21`.

Installation
------------

See the `"Installation" section in the README file on GitHub <https://github.com/For-a-few-DPPs-more/structure-factor#installation>`_.

Getting started
---------------

Please refer to

- the tutorial `Jupyter notebook <https://github.com/For-a-few-DPPs-more/structure-factor/tree/main/notebooks>`_, and
- the different sections of the present documentation.

Contents
--------

.. toctree::
   :maxdepth: 3

   hyperuniformity
   multiscale_estimators
   pair_correlation_function
   point_pattern
   point_processes
   spatial_windows
   structure_factor
   tapers
   tapered_estimators
   tapered_estimators_isotropic
   transforms
   utils
   plotting
   bibliography/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
