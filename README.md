# structure_factor:

Approximate the structure facture of a point pattern (a stationary point process) of \\(\mathbb{R}^d \\), and test the effective hyperuniformity and the class of hyperuniformity of the point pattern.

## Introduction:
 The structure factor \\( S(\mathbf{k}) \\)of a stationary point process of  \\( \mathbb{R}^d \\) with inetnsity \\( \rho \\) is defined (when it exists) through the Fourier transform of its pair correlation function \\( g \\) by,
 \\[ S(\mathbf{k}) = 1 + \rho \mathcal{F}(g -1)(\mathbf{k})\\].
 In condensed matter physics, it has been observed for some particle systems that, the variance of the number of points in a large window is lower than expected, a phenomenon called hyperuniformity.
 A point process is hyperunifrom if
 \\[ \lim_{\| \mathbf{k}\| \rightarrow 0} S(\mathbf{k}) = 0 . \\]
 Also the call of hyperuniformity is related to the power decay of the structure factor near 0.
 Structure_factor is a Python library that gathers 3 estimators of the structure factor:
    1. The scattering inetnsity \\( \widehat{S}_{SI} \\).
    2. An estimator using [Ogata quadrature](https://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf) for approximating the Hankel transform \\( \widehat{S}_{HO} \\).
    3. An estimator using [Baddour and Chouinard Discrete Hankel transform](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-32-4-611) \\( \widehat{S}_{HBC} \\).

  It also contains a test of effective hyperunifomity using the index H and a test of the decay of the structure factor near 0 to estimate the class of hyperuniformity.


 ## Installation
- structure_factor works with [Python 3.8+](https://www.python.org/downloads/release/python-380/).
- structure_factor is now available on `PyPI <https://pypi.org/project/>`__ |PyPI package|
- .. code:: bash

  pip install structure_factor

All the necessary dependency will be automatically installed.

### Requirements

Currently, the project calls the `spatstat` R package

- [R programming language](https://www.r-project.org/)
- project dependencies, see [`tool.poetry.dependencies` in `pyproject.toml`](./pyproject.toml)

## How to use it

TBC

## Documentation

### Locally

We build the documentation with [Sphinx](https://www.sphinx-doc.org/en/master/index.html) from `.rst` (reStructuredText) files.

- [Sphinx documentation](https://www.sphinx-doc.org/en/master/index.html)
- [rst cheatsheet](https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/CheatSheet.html
)
- Examples
  - [DPPy](https://github.com/guilgautier/DPPy) and  [online documentation](https://dppy.readthedocs.io/en/latest/?badge=latest)
  - [rlberry-py](https://github.com/rlberry-py/rlberry) and [online documentation](https://rlberry.readthedocs.io/en/latest/?badge=latest)

### Remotely

The documentation will be built by and published on [ReadTheDocs](https://readthedocs.org/)

## How to cite this work?

TBC
