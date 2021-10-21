# structure_factor

*''Approximate the structure facture of a point pattern (a stationary point process), and test the effective hyperuniformity and the class of hyperuniformity of the point pattern"*.

- [structure_factor](#structure_factor)
  - [Introduction](#introduction)
  - [Related paper](#related-paper)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [How to use it](#how-to-use-it)
  - [How to cite this work](#how-to-cite-this-work)
  - [Reproductibility](#reproductibility)

## Introduction

 In condensed matter physics, it has been observed for some particle systems that, the variance of the number of points in a large window is lower than expected, a phenomenon called hyperuniformity.
 A point process is also called hyperunifrom if its structure factor  (defined through the Fourier transform of its pair correlation function) vanishes on zero.

 Moreover, the repartition of hyperuniform point processes between classes is related to the power decay of the structure factor near 0.

 `structure_factor` is a Python library that gathers:

- Three estimators of the structure factor:
    1. The scattering inetnsity .
    2. An estimator using [Ogata quadrature](https://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf) for approximating the Hankel transform.
    3. An estimator using [Baddour and Chouinard Discrete Hankel transform](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-32-4-611).

- Two estimators of the pair correlation function:
   1. Estimator using Epanechnikov kernel and a bandwidth selected by Stoyan's rule of thumb.
   2. Estimator using the derivative of Ripley's K function.

  This 2 estimators are obtained using [spatstat-interface](https://github.com/For-a-few-DPPs-more/spatstat-interface) which builds a hidden interface with the package [`spatstat`](https://github.com/spatstat/spatstat) of the programming language [R](https://www.r-project.org/),

- Two tests of hyperuniformity:

  1. Test of effective hyperunifomity using the index H of hyperunifomity.
  2. Test of the decay of the structure factor near 0 to estimate the class of hyperuniformity.

## Related paper

We wrote a companion paper to `structure_factor`, tilted "Exploring the hyperuniformity of a point process using structure_factor" (which will be published soon) explaining in detail the mathematical foundations behind the estimators present in `structure_factor`. This paper also contains detailed tests of this package on 3 different point processes and discuss the results and  the limitations of the package.

## Dependencies

Currently, the project calls the [`spatstat`](https://github.com/spatstat/spatstat) [R](https://www.r-project.org/) package

- [R programming language](https://www.r-project.org/)
- project dependencies, see [`tool.poetry.dependencies` in `pyproject.toml`](./pyproject.toml)

Note that all the necessary **project dependencies** will be automatically installed.

## Installation

- structure_factor works with [Python 3.8+](https://www.python.org/downloads/release/python-380/).

- Installation using [PyPI](https://pypi.org/project/).

      pip install structure_factor

- Installation in editable mode and potentially contribute to the project
  - You may consider [forking the repository](https://github.com/For-a-few-DPPs-more/structure_factor/fork).
  - In any case, your can clone the repository
     1. if you have forked the repository

      ```bash
      git clone https://github.com/your_user_name/structure_factor.git
      ```

     2. if you have **not** forked the repository

      ```bash
      git clone https://github.com/For-a-few-DPPs-more/structure_factor.git
  - Installation using `poetry`

    The package can be installed in **editable** mode along with

    - main (non-optional) dependencies, see `[tool.poetry.dependencies]` in [`pyproject.toml`](./pyproject.toml)
    - development dependencies, `[tool.poetry.dev-dependencies]` in [`pyproject.toml`](./pyproject.toml)

    ```bash
    cd structure_factor
    # activate your virtual environment or run
    # poetry shell  # to create/activate local .venv (see poetry.toml)
    poetry install
    # poetry install --no-dev  # to avoid installing the development dependencies
    ```

## Documentation

- We build the documentation with [Sphinx](https://www.sphinx-doc.org/en/master/index.html) from `.rst` (reStructuredText) files.

  - [Sphinx documentation](https://www.sphinx-doc.org/en/master/index.html)
  - [rst cheatsheet](https://docs.typo3.org/m/typo3/docs-how-to-document/master/en-us/WritingReST/CheatSheet.html
)
- The documentation will be built by and published on [ReadTheDocs](https://readthedocs.org/)
- If you wish to contribute to the documentation or just having it locally, you can:
  - Generate the docs locally

    ```bash
     cd structure_factor/docs
     make html

  - Open the local HTML version of the documentation

    ```bash
    open _build/html/index.html

## How to use it

A tutorial in Jupyter notebook is available (...). You can read and work on the interactive tutorial Notebook (add link to the notebook), directly from your web browser, without having to download or install Python or anything. Just click, wait a little bit, and play with the notebook!

## How to cite this work

If you use the structure_factor toolbox, please consider citing it with this piece of BibTeX:
TBC

## Reproductibility

TBC
