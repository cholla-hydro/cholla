# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import functools
import json
import os
import subprocess
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
    'breathe',
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
    "par"
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
    'gh-issue' : (_GITHUB_BASE + '/issues/%s', 'gh-issue#%s'),
    'gh-pr' : (_GITHUB_BASE + '/pull/%s', 'gh-pr#%s')
}
# As an example, if you write
#   {repository-file}`builds/make.type.hydro`
# it should be converted to a link (displaying the specified path) that links
# to the GitHub Page for make.type.hydro

# -- Doxygen/Breathe Stuff ---------------------------------------------------

def _it_tree_paths(dir_path, include_dirs=True):
    # recursive iterate over paths to all files (and possibly directories)
    for root, dirs, files in os.walk(dir_path, followlinks=False):
        if include_dirs:
            yield from (os.path.join(root, d) for d in dirs)
        yield from (os.path.join(root, f) for f in files)

def _dirtree_mtime(dir_path):
    """
    Walk a directory and determine the most recent time at which a contained
    file/directory was modified, created, deleted, removed, etc.
    """

    def get_mtime(path): # gives posix timestamp in seconds (rounded up)
        return os.stat(dir_path).st_mtime + 1

    # explicitly measure mtime of root dir
    root_mtime = get_mtime(dir_path)

    # make iterator over the mtimes of each item in dir_path. We explicitly
    # check mtimes of directories since they provide the only indication that
    # files within a that directory were deleted/moved
    itr = (get_mtime(p) for p in _it_tree_paths(dir_path, include_dirs=True))

    return functools.reduce(max, itr, root_mtime)

def build_doxygen():
    if os.getenv("SKIPDOXYGEN", "FALSE").lower() == "true":
        return None # skip a rebuild

    # load cached modification times (if they exist)
    try:
        with open("../cached_mtimes.json", "r") as f:
            cached_mtimes = json.load(f)
    except FileNotFoundError:
        cached_mtimes = {"src" : None, "dox" : None}

    dox_builddir = "../doxygen/build/xml/"

    # determine modification time of the source code directory tree
    src_mtime = _dirtree_mtime("../../src")

    # if the doxygen build-dir already exists, and the modification times
    # (of the source code directory and the doxygen build-dir) match the
    # cached values, then we don't need to regenerate the documentation
    if (
        os.path.isdir(dox_builddir) and
        (cached_mtimes["src"] == src_mtime) and
        (cached_mtimes["dox"] == _dirtree_mtime(dox_builddir))
    ):
        return None

    try:
        retcode = subprocess.call("doxygen", cwd="../doxygen")

        if retcode < 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
        else:
            # get modification time of dox_builddir (after the build)
            dox_mtime = _dirtree_mtime(dox_builddir)
            mtime_pack = {"src": src_mtime, "dox": dox_mtime}
            with open("../cached_mtimes.json", "w") as f:
                json.dump(mtime_pack, f)
    except OSError as e:
        sys.stderr.write(f"doxygen execution failed: {e}")

    return {"cholla": dox_builddir}

_header_files = [
    os.path.abspath(p) for p in _it_tree_paths("../../src") if p.endswith(".h")
]
breath_projects = build_doxygen() # always build doxygen docs (for local & rtd builds)
breathe_default_project = "cholla"
breathe_projects_source = {"cholla" : ( "../../src/", _header_files )}
