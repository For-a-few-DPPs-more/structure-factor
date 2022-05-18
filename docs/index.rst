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

Hyperuniformity is the study of stationary point processes with a sub-Poisson variance of the number of points in a large window. For the homogeneous Poisson point process, the variance of the number of points that fall in a large window is of the order of the window volume. In contrast, for hyperuniform (HU) point processes, the corresponding variance is much lower than the volume of that window, with a ratio going to zero.

Hyperuniform point processes have received a lot of attention in statistical physics, both for the investigation of natural organized structures and the synthesis of materials.
There are many candidate HU processes in the physics literature, but rigorously proving that a point process is HU is usually difficult. It is thus desirable to have standardized numerical tests of hyperuniformity. A common practice in statistical physics and chemistry is to use a few samples to estimate a spectral measure called the structure factor, and evaluating its decay around zero provides a diagnostic of hyperuniformity.
Different applied fields use however different estimators, and important algorithmic choices proceed from each field's lore.

In an effort to make investigations of the structure factor and hyperuniformity systematic and reproducible, we further provide
this Python toolbox gathering the estimators of the structure factor surveyed in :cite:`HGBLR:22`, along with a new statistical test of hyperuniformity asymptotically valid :cite:`HGBLR:22`, the test of effective hyperuniformity :cite:`Tor18` and the corresponding possible class of hyperuniformity :cite:`Cos21`.

Companion paper
---------------
We wrote a companion paper to ``structure-factor``,

   "`On estimating the structure factor of a point process, with applications to hyperuniformity <https://arxiv.org/abs/2203.08749>`_ ".

where we provided rigorous mathematical derivations of the structure factor's estimators of a stationary point process and showcased ``structure-factor`` on different point processes.
We also contribute a new asymptotically valid statistical test of hyperuniformity.
Finally, we compared numerically the accuracy of the estimators.

If you use ``structure-factor``, please consider citing the companion paper.

Installation
--------------

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
