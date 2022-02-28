# structure-factor

[![CI-tests](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/ci.yml/badge.svg)](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/For-a-few-DPPs-more/structure-factor/branch/main/graph/badge.svg?token=FUDADJLO2W)](https://codecov.io/gh/For-a-few-DPPs-more/structure-factor)
[![docs-build](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/docs.yml/badge.svg)](https://github.com/For-a-few-DPPs-more/structure-factor/actions/workflows/docs.yml)
[![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://for-a-few-dpps-more.github.io/structure-factor/)
[![PyPi version](https://badgen.net/pypi/v/structure-factor/)](https://pypi.org/project/structure-factor/)
[![Python >=3.7.1,<3.10](https://img.shields.io/badge/python->=3.7.1,<3.10-blue.svg)](https://www.python.org/downloads/release/python-371/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks)

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
    - [Documentation](#documentation-1)
    - [Notebooks](#notebooks)

## Introduction

`structure-factor` is an open-source Python project which currently collects

- various estimators of the structure factor,
- and several diagnostics of hyperuniformity,

for stationary and isotropic point processes.

Please checkout the [documentation](https://for-a-few-dpps-more.github.io/structure-factor/) for more details.

## Dependencies

- [R programming language](https://www.r-project.org/), since we call the [`spatstat`](https://github.com/spatstat/spatstat) R package to estimate the pair correlation function of point processes using [`spatstat-interface`](https://github.com/For-a-few-DPPs-more/spatstat-interface).

- Python dependencies are listed in the [`pyproject.toml`](./pyproject.toml) file.

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
  # poetry add structure-factor@latest to update if already present
  # pip install --upgrade structure-factor
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
  # activate a virtual environment
  pip install '.[notebook]'  # (see [tool.poetry.extras] in pyproject.toml)
  sphinx-build -b html docs docs/_build/html
  open _build/html/index.html
  ```

## Getting started

### Documentation

See the documentation [![docs-page](https://img.shields.io/badge/docs-latest-blue)](https://for-a-few-dpps-more.github.io/structure-factor/)

### Notebooks

[Jupyter](https://jupyter.org/) that showcase `structure-factor` are available in the [./notebooks](./notebooks) folder.

<!--
## How to cite this work

### Companion paper

A companion paper is being written

> Exploring the hyperuniformity of a point process using structure-factor

where we provide rigorous mathematical derivations of the different estimators of the structure factor and showcase `structure-factor` on three different point processes.

If you use `structure-factor`, please consider citing it with this piece of BibTeX:

  ``` -->
