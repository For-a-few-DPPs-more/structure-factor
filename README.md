# structure-factor

[![CI-tests](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/ci.yml/badge.svg)](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/For-a-few-DPPs-more/structure-factor/branch/main/graph/badge.svg?token=FUDADJLO2W)](https://codecov.io/gh/For-a-few-DPPs-more/structure-factor)
[![docs-build](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/docs.yml/badge.svg)](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/docs.yml)
[![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://for-a-few-dpps-more.github.io/structure-factor/)
[![PyPi version](https://badgen.net/pypi/v/structure-factor/)](https://pypi.org/project/structure-factor/)
[![Python >=3.7.1,<3.10](https://img.shields.io/badge/python->=3.7.1,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/)

> Approximate the structure factor of a stationary point process, test its effective hyperuniformity, and identify its class of hyperuniformity.

- [structure-factor](#structure-factor)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
    - [Install the project as a dependency](#install-the-project-as-a-dependency)
    - [Install in editable mode and potentially contribute to the project](#install-in-editable-mode-and-potentially-contribute-to-the-project)
  - [Documentation](#documentation)
    - [Build the documentation](#build-the-documentation)
  - [Getting started](#getting-started)

## Introduction

Hyperuniformity is the study of stationary point processes with a sub-Poisson variance of the number of points in a large window.
To study the hyperuniformity of a given point process, a common practice in statistical physics is to estimate a spectral measure called the **structure factor**, the behavior of which around zero is a sign of hyperuniformity. The structure factor of a point process is defined via the Fourier transform of its total correlation function, and a point process is hyperuniform if its structure factor vanishes at zero.
This Python toolbox gathers many estimators of the structure factor, along with a numerical test of effective hyperuniformity and a test for identifying the possible hyperuniformity class.

---

`structure-factor` is an open-source Python project which currently collects

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
  python = ">=3.7.1,<3.10"

  numpy = "^1.20.3"
  scipy = "^1.6.3"
  matplotlib = "^3.4.2"
  pandas = "^1.2.4"
  spatstat-interface = "^0.1.0"
  # spatstat-interface https://github.com/For-a-few-DPPs-more/spatstat-interface requires rpy2 https://rpy2.github.io/
  ```

## Installation

`structure-factor` works with [![Python >=3.7.1,<3.10](https://img.shields.io/badge/python->=3.7.1,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/).

Once installed it can be called from

- `import structure_factor`
- `from structure_factor import ...`

### Install the project as a dependency

- Install the latest version published on [![PyPi version](https://badgen.net/pypi/v/structure-factor/)](https://pypi.org/project/structure-factor/)

  ```bash
  # activate your virtual environment an run
  poetry add structure-factor
  # pip install structure-factor
  ```

- Install from source (this may be broken)

  ```bash
  # activate your virtual environment and run
  poetry add git+https://github.com/For-a-few-DPPs-more/structure-factor.git
  # pip install git+https://github.com/For-a-few-DPPs-more/structure-factor.git
  ```

### Install in editable mode and potentially contribute to the project

The package can be installed in **editable** mode using [`poetry`](https://python-poetry.org/).

To do this, clone the repository:

- if you considered [forking the repository](https://github.com/For-a-few-DPPs-more/structure-factor/fork)

  ```bash
  git clone https://github.com/your_user_name/structure-factor.git
  ```

- if you have **not** forked the repository

  ```bash
  git clone https://github.com/For-a-few-DPPs-more/structure-factor.git
  ```

and install the package in editable mode

```bash
cd structure-factor
poetry shell  # to create/activate local .venv (see poetry.toml)
poetry install
# poetry install --no-dev  # to avoid installing the development dependencies
# poetry add -E docs -E notebook  # to install extra dependencies
```

## Documentation

The documentation [![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://for-a-few-dpps-more.github.io/structure-factor/) is

- generated using [Sphinx](https://www.sphinx-doc.org/en/master/index.html), and
- published via the GitHub workflow file [.github/workflows/docs.yml](.github/workflows/docs.yml).

### Build the documentation

If you use `poetry`

- install the documentation dependencies (see `[tool.poetry.extras]` in [`pyproject.toml`](./pyproject.toml))

  ```bash
  cd structure-factor
  poetry shell  # to create/activate local .venv (see poetry.toml)
  poetry install -E docs  # (see [tool.poetry.extras] in pyproject.toml)
  ```

- and run

  ```bash
  # cd structure-factor
  # poetry shell  # to create/activate local .venv (see poetry.toml)
  poetry run sphinx-build -b html docs docs/_build/html
  open _build/html/index.html
  ```

Otherwise, if you don't use `poetry`

- install the documentation dependencies (listed in `[tool.poetry.extras]` in [`pyproject.toml`](./pyproject.toml)), and

- run

  ```bash
  cd structure-factor
  sphinx-build -b html docs docs/_build/html
  open _build/html/index.html
  ```

## Getting started

- [Jupyter](https://jupyter.org/) notebooks that showcase `structure-factor` are available in the [./notebooks](./notebooks) folder.

  - if you use `poetry`

    ```bash
    cd structure-factor
    poetry shell  # to create/activate local .venv (see poetry.toml)
    poetry install -E notebook  # (see [tool.poetry.extras] in pyproject.toml)
    # open a notebook within VSCode for example
    ```

- See the documentation [![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://for-a-few-dpps-more.github.io/structure-factor/)

<!--
## How to cite this work

### Companion paper

A companion paper is being written

> Exploring the hyperuniformity of a point process using structure-factor

where we provide rigorous mathematical derivations of the different estimators of the structure factor and showcase `structure-factor` on three different point processes.

If you use `structure-factor`, please consider citing it with this piece of BibTeX:

  ``` -->
