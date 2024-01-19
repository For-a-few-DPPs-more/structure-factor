# Notebooks

<!-- ## Run a notebook remotely

- `tutorial_structure_factor.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/For-a-few-DPPs-more/structure-factor/blob/main/notebooks/tutorial_structure_factor.ipynb)
-->
## Run a notebook locally

It is suggested you clone the repository and get the latest version of the source code

```bash
git clone https://github.com/For-a-few-DPPs-more/structure-factor.git
cd structure-factor
git pull origin main
```

Then, install the project in a virtual environement and launch the notebook

- if you use [`poetry`](https://python-poetry.org/)

    ```bash
    poetry shell  # to create/activate local .venv (see poetry.toml)
    poetry install -E notebook  # (see [tool.poetry.extras] in pyproject.toml)
    # open a notebook within VSCode for example
    # or run
    # poetry run jupyter notebook
    ```

- if you use `pip`

    ```bash
    # activate a virtual environment of your choice and run
    pip install '.[notebook]'  # (see [tool.poetry.extras] in pyproject.toml)
    # open a notebook within VSCode for example
    # or run
    # jupyter notebook
    ```
