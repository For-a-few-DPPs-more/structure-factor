# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))


# -- Project information -----------------------------------------------------

project = "structure-factor"
copyright = "2021-2022, Diala Hawat"
author = "Diala Hawat, Guillaume Gautier, Rémi Bardenet, and Raphael Lachieze-Rey"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Include documentation from docstrings
    "sphinx.ext.coverage",  # Collect doc coverage stats
    "sphinx.ext.doctest",  # Test snippets in the documentation
    "sphinx.ext.githubpages",  # Publish HTML docs in GitHub Pages
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.todo",  # .. todo:: directive
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinxcontrib.bibtex",  # Sphinx extension for BibTeX style citations
    "sphinxcontrib.proof",  # Sphinx extension to typeset theorems, proofs
    "matplotlib.sphinxext.plot_directive",  # .. plot:: directive for matplotlib plot
]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "body_max_width": "100%",
    "navigation_depth": 4,
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
html_context = {
    "display_github": True,  # Integrate Github
    "github_user": "For-a-few-DPPs-more",  # Username
    "github_repo": "structure-factor",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# member order
autodoc_member_order = "bysource"


# -- Extension configuration -------------------------------------------------

# sphinxcontrib-bibtex
# https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html
bibtex_bibfiles = ["./bibliography/bibliography.bib"]
bibtex_encoding = "latin"
# bibtex_reference_style = "alpha"  # alpha, plain , unsrt, and unsrtalpha

# matplotlib.sphinxext.plot_directive
# https://matplotlib.org/stable/api/sphinxext_plot_directive_api.html

plot_include_source = True
# plot_html_show_source_link =
# plot_pre_code =
# plot_basedir =
# plot_formats =
# plot_html_show_formats =
# plot_rcparams =
# plot_apply_rcparams =
# plot_working_directory =
# plot_template =
