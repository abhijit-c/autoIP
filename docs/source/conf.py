# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "autoIP"
copyright = "2023, Abhijit Chowdhary"
author = "Abhijit Chowdhary"
release = "0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sys, os

# sys.path.append("../../")
sys.path.insert(0, os.path.abspath("../.."))
import autoip


autosummary_generate = True  # Make _autosummary files and include them

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.inheritance_diagram",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_type_aliases = {
    "ArrayLike": "ArrayLike",
    "Array": "Array",
    "LinearOperator": "LinearOperator",
    "Operator": "Operator",
    "PRNGKey": "PRNGKey",
    # "LinearOperator": "Callable[[ArrayLike], Array]",
    # "Operator": "Callable[[ArrayLike], Array]",
}

autodoc_typehints = "description"

autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
