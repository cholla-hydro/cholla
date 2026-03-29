# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('./_ext'))

# -- Project information -----------------------------------------------------

project = 'Cholla'
copyright = '2025, cholla developers'
#author = 'Nope'

# The full version, including alpha/beta/rc tags
release = '3.0.1-dev'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinxcontrib.video',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    #'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.extlinks',
    'sphinx_inline_tabs',
    'sphinx_autodoc_typehints', # <- need to be loaded after napoleon
    # the following was all used in an earlier version (but don't seem necessary)
    #'sphinx.ext.doctest',
    #'sphinx.ext.imgmath',
    #'sphinx.ext.mathjax', # <- how does this improve math rendering?

    # Custom Extensions
    # -----------------
    "doxybuild",
    "par",
    "cli_help",
]



source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["Reference/param/**"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = {"par_formatting.css"}

# Extension Options
# =================

todo_include_todos = True

autosummary_generate = ["Reference/PythonApiRef.rst"]

# -- Options for sphinx.ext.napoleon -----------------------------------------
# (for rendering python docstrings)
napoleon_google_docstring = False
napoleon_use_rtype = True

# -- Options for MyST --------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "fieldlist"
]

# -- Options for doxybuild extension -----------------------------------------

# path to the baseline doxyfile (relative to this config file)
doxybuild_hardcoded_doxyfile = "../doxygen/Doxyfile"
# path to the C++ source code directory (relative to this config file)
doxybuild_src_code_dir = "../../src"
# path relative to the source directory where the stub files are written
doxybuild_dest_dir = "Reference/internal-api-ref"
# override doxygen parameters https://www.doxygen.nl/manual/config.html
doxybuild_overrides = {"PROJECT_NUMBER": release}

# -- Options for par extension -----------------------------------------------

par_separator = "."

# -- Options for sphinx.ext.extlinks -----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/extlinks.html#module-sphinx.ext.extlinks

# This config is a dictionary of external sites, where the key is used as a
# name of a role and the value is the name of a tuple of strings that serve as
# templates for an external url and a template for the text that gets used.
_GITHUB_BASE = 'https://github.com/cholla-hydro/cholla'
extlinks = {
    'repository-file': (f'{_GITHUB_BASE}/tree/dev/' + '%s', '%s'),
    # repository-dir indeed looks the same as repository file, but it makes
    # sense to differentiate (just in case we ever want to do more
    # sophisticated formatting or move away from github)
    'repository-dir': (f'{_GITHUB_BASE}/tree/dev/' + '%s', '%s'),
    'gh-issue' : (_GITHUB_BASE + '/issues/%s', 'gh-issue#%s'),
    'gh-pr' : (_GITHUB_BASE + '/pull/%s', 'gh-pr#%s')
}
# As an example, if you write
#   {repository-file}`config/make.type.hydro`
# it should be converted to a link (displaying the specified path) that links
# to the GitHub Page for make.type.hydro

# -- Doxygen/Breathe Stuff ---------------------------------------------------

