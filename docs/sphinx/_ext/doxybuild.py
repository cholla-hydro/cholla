"""
Build doxygen documentation.
"""

import functools
import json
import os
import shutil
import subprocess
import sys


def _it_tree_paths(dir_path, include_dirs=True):
    # recursive iterate over paths to all files (and possibly directories)
    for root, dirs, files in os.walk(dir_path, followlinks=False):
        if include_dirs:
            yield from (os.path.join(root, d) for d in dirs)
        yield from (os.path.join(root, f) for f in files)


def _get_mtime(nominal_path):
    """
    Walk a directory and determine the most recent time at which a contained
    file/directory was modified, created, deleted, removed, etc.
    """

    def _fn(path):  # gives posix timestamp in seconds (rounded up)
        return os.stat(path).st_mtime + 1

    # explicitly measure mtime of nominal_path
    root_mtime = _fn(nominal_path)

    if os.path.isfile(nominal_path):
        return root_mtime

    # make iterator over the mtimes of each item in dir_path. We explicitly
    # check mtimes of directories since they provide the only indication that
    # files within a that directory were deleted/moved
    itr = (_fn(p) for p in _it_tree_paths(nominal_path, include_dirs=True))

    return functools.reduce(max, itr, root_mtime)


def build_doxygen():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", ".."))
    src_dir = os.path.join(root_dir, "src")
    doxygen_dir = os.path.join(root_dir, "docs", "doxygen")
    doxyfile = os.path.join(doxygen_dir, "Doxyfile")

    dox_builddir = os.path.join(doxygen_dir, "build", "html")

    _header_files = [
        os.path.abspath(p) for p in _it_tree_paths(src_dir) if p.endswith(".h")
    ]

    # load cached modification times (if they exist)
    try:
        with open(os.path.join(root_dir, "docs/cached_mtimes.json"), "r") as f:
            cached_mtimes = json.load(f)
    except FileNotFoundError:
        cached_mtimes = {}

    # determine modification time of the source code directory tree
    src_mtime = _get_mtime(src_dir)
    doxyfile_mtime = _get_mtime(doxyfile)

    # if the doxygen build-dir already exists, and the modification times
    # (of the source code directory and the doxygen build-dir) match the
    # cached values, then we don't need to regenerate the documentation
    if (
        (cached_mtimes.get("src", None) == src_mtime)
        and (cached_mtimes.get("doxyfile", None) == doxyfile_mtime)
        and os.path.isdir(dox_builddir)
        and (cached_mtimes.get("dox-build-dir", None) == _get_mtime(dox_builddir))
    ):
        return dox_builddir

    if os.path.exists(dox_builddir):
        shutil.rmtree(dox_builddir)

    try:
        retcode = subprocess.call(["doxygen", doxyfile], cwd=doxygen_dir)

        if retcode != 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
            if os.path.exists(dox_builddir):
                shutil.rmtree(dox_builddir)
        else:
            # get modification time of dox_builddir (after the build)
            dox_build_dir_mtime = _get_mtime(dox_builddir)
            mtime_pack = {
                "src": src_mtime,
                "dox-build-dir": dox_build_dir_mtime,
                "doxyfile": doxyfile_mtime,
            }
            with open("../cached_mtimes.json", "w") as f:
                json.dump(mtime_pack, f)
    except OSError as e:
        sys.stderr.write(f"doxygen execution failed: {e}")

    return dox_builddir


def generate_and_copy_doxygen(sphinx_build_dir):
    # a short-term hack to get the build-directory
    # -> the proper fix involves making a sphinx extension
    dox_builddir = build_doxygen()

    # copy the files into the builddir
    # TODO: build them at the destination to begin with
    dox_dest_builddir = os.path.join(sphinx_build_dir, "internal-api-ref")
    os.makedirs(dox_dest_builddir, exist_ok=True)
    shutil.copytree(src=dox_builddir, dst=dox_dest_builddir, dirs_exist_ok=True)


def setup(app):
    outdir = app.outdir
    assert outdir.endswith("html")

    if os.getenv("SKIPDOXYGEN", "FALSE").lower() == "true":
        pass  # do nothing!
    else:
        generate_and_copy_doxygen(outdir)
