.. structure-factor documentation master file, created by
   sphinx-quickstart on Wed Jul 21 15:25:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to structure_factor's documentation:
============================================

This is the documentation of the package ``structure_factor`` which is an open source Python toolbox available on `Github <https://github.com/For-a-few-DPPs-more/structure-factor>`_. Moreover, a `tutorial Jupyter Notebook <https://github.com/For-a-few-DPPs-more/structure-factor/tree/main/notebooks>`_ is available, which could be very useful for the users of this package.

**Introduction:**

In condensed matter physics, it has been observed for some particle systems that, the variance of the number of points in a large window is lower than expected, a phenomenon called hyperuniformity.
To study the hyperuniformity of a given point process, common practice in statistical physics is to estimate a spectral measure called the **structure factor**, the behavior of which around zero is a sign of hyperuniformity. The structure factor of a point process is defined via the Fourier transform of its total correlation function. For more details [see](https://scoste.fr/assets/survey_hyperuniformity.pdf).

The package ``structure_factor`` is designed to approximate the structure factor of a stationary point process, test its effective hyperuniformity and identify its class of hyperuniformity.

**Contents:**

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
