# structure_factor

> Approximate the structure factor of a stationary point process and test its effective hyperuniformity and identify its class of hyperuniformity.

- [structure_factor](#structure_factor)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
    - [Install the project as a dependency](#install-the-project-as-a-dependency)
    - [Install in editable mode and potentially contribute to the project](#install-in-editable-mode-and-potentially-contribute-to-the-project)
  - [Documentation](#documentation)
    - [Build the documentation](#build-the-documentation)
  - [Getting started](#getting-started)
  - [How to cite this work](#how-to-cite-this-work)
    - [Companion paper](#companion-paper)

## Introduction

In condensed matter physics, it has been observed for some particle systems that, the variance of the number of points in a large window is lower than expected, a phenomenon called hyperuniformity.

TODO: Introduce briefly the structure factor: observable/measurable, defined through the Fourier transform of the pair correlation function.

A point process is also called hyperuniform if its structure factor vanishes at zero.

Classes of hyperuniform point processes are then defined based on the decay of the structure factor around zero.

---

The Python package `structure_factor` currently collects

- Three estimators of the structure factor:
    1. the scattering intensity,
    2. an estimator using [Ogata quadrature](https://www.kurims.kyoto-u.ac.jp/~prims/pdf/41-4/41-4-40.pdf) for approximating the Hankel transform,
    3. an estimator using [Baddour and Chouinard Discrete Hankel transform](https://www.osapublishing.org/josaa/abstract.cfm?uri=josaa-32-4-611).

- Two ways to qualify hyperuniformity:

  1. effective hyperuniformity,
  2. classes of hyperuniformity.

## Dependencies

- [R programming language](https://www.r-project.org/), since we call the [`spatstat`](https://github.com/spatstat/spatstat) R package to estimate the pair correlation function of point processes using [`spatstat-interface`](https://github.com/For-a-few-DPPs-more/spatstat-interface).

- Python dependencies are listed in the [`pyproject.toml`](./pyproject.toml) file. Note that they mostly correspond to the latest version.

  ```toml
  [tool.poetry.dependencies]
  python = ">=3.8,<3.10"

  numpy = "^1.20.3"
  scipy = "^1.6.3"
  matplotlib = "^3.4.2"
  pandas = "^1.2.4"
  spatstat-interface = "^0.1.0"
  # spatstat-interface https://github.com/For-a-few-DPPs-more/spatstat-interface requires rpy2 https://rpy2.github.io/
  ```

## Installation

`structure_factor` works with [Python 3.8+](https://www.python.org/downloads/release/python-380/).

### Install the project as a dependency

<!-- - Install the latest version published on [![PyPi version](https://badgen.net/pypi/v/structure_factor/)](https://pypi.org/project/structure_factor/)

  ```bash
  # activate your virtual environment an run
  poetry add structure_factor
  # pip install structure_factor
  ``` -->

- Install from source (this may be broken)

  ```bash
  # activate your virtual environment and run
  poetry add git+https://github.com/For-a-few-DPPs-more/structure_factor.git
  # pip install git+https://github.com/For-a-few-DPPs-more/structure_factor.git
  ```

### Install in editable mode and potentially contribute to the project

The package can be installed in **editable** mode using [`poetry`](https://python-poetry.org/).

To to this, clone the repository:

- if you considered [forking the repository](https://github.com/For-a-few-DPPs-more/structure_factor/fork)

  ```bash
  git clone https://github.com/your_user_name/structure_factor.git
  ```

- if you have **not** forked the repository

  ```bash
  git clone https://github.com/For-a-few-DPPs-more/structure_factor.git
  ```

and install the package in editable mode

```bash
cd structure_factor
# activate your virtual environment and run
# poetry shell  # to create/activate local .venv (see poetry.toml)
poetry install
# poetry install --no-dev  # to avoid installing the development dependencies
# poetry add -E docs -E notebook  # to install extra dependencies
```

## Documentation

The documentation <https://for-a-few-dpps-more.github.io/structure-factor> is

- generated using [Sphinx](https://www.sphinx-doc.org/en/master/index.html), and
- published via the GitHub workflow file [.github/workflows/docs.yml](.github/workflows/docs.yml).

### Build the documentation

Assuming `structure_factor` has been installed, you can simply run

```bash
  # activate your virtual environment
  cd structure_factor
  sphinx-build -b html docs docs/_build/html
  # poetry run sphinx-build -b html docs docs/_build/html
  open _build/html/index.html
```

## Getting started

- [Jupyter](https://jupyter.org/) notebooks that showcase `structure_factor` are available in the [./notebooks](./notebooks) folder.
- See the documentation <https://for-a-few-dpps-more.github.io/structure-factor>

## How to cite this work

### Companion paper

A companion paper is being written

> Exploring the hyperuniformity of a point process using structure_factor

where we provide rigorous mathematical derivations of the different estimators of the structure factor and showcase `structure_factor` on three different point processes.

If you use the `structure_factor` package, please consider citing it with this piece of BibTeX:

TBC
