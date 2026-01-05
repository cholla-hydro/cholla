"""
Build doxygen documentation.
"""
from collections.abc import Mapping, Iterator
import functools
import json
import os
import shutil
import string
import subprocess
import sys
import typing

def _it_tree_paths(dir_path: os.PathLike, include_dirs: bool=True)->Iterator[str]:
    # recursive iterate over paths to all files (and possibly directories)
    for root, dirs, files in os.walk(dir_path, followlinks=False):
        if include_dirs:
            yield from (str(os.path.join(root, d)) for d in dirs)
        yield from (str(os.path.join(root, f)) for f in files)


def _get_mtime(nominal_path: os.PathLike) -> int:
    """
    Walk a directory and determine the most recent time at which a contained
    file/directory was modified, created, deleted, removed, etc.
    """

    def _fn(path):  # gives posix timestamp in seconds (rounded up)
        return int(os.stat(path).st_mtime + 1)

    # explicitly measure mtime of nominal_path
    root_mtime = _fn(nominal_path)

    if os.path.isfile(nominal_path):
        return root_mtime

    # make iterator over the mtimes of each item in dir_path. We explicitly
    # check mtimes of directories since they provide the only indication that
    # files within a that directory were deleted/moved
    itr = (_fn(p) for p in _it_tree_paths(nominal_path, include_dirs=True))

    return functools.reduce(max, itr, root_mtime)


_TEMPLATE = """
# this is a template for a Doxyfile that can be used by Sphinx
# -> the basic premise is to leave the original Doxygen file with sensible
#    defaults (so that we can build the docs without sphinx)
# -> this file will pull in those settings and selectively overwrite them

@INCLUDE = {RAW_CONF_FILE}

OUTPUT_DIRECTORY       = {OUTPUT_DIRECTORY}
INPUT                  = {INPUT}
"""
#PROJECT_NUMBER         = {DOXYGEN_VERSION_STRING}
#DOT_PATH               = {DOXYGEN_DOT_PATH}

def _write_template(f: typing.IO, template: str, mapping: Mapping[str,str]):
    # get the field names in the string
    field_name_set = set(
        quad[1] for quad in string.Formatter().parse(template) if quad[1] is not None
    )
    if len(field_name_set.symmetric_difference(mapping)) != 0:
        for name in field_name_set:
            if name not in mapping:
                raise ValueError(f"{name!r} was not provided by the mapping")
        for name in mapping:
            if name not in field_name_set:
                raise ValueError(
                    f"{name!r} was provided by the mapping, but not known to template"
                )
    else:
        f.write(template.format_map(mapping))


def _try_make_mtime_cache(cache_keys: dict[str, os.PathLike]) -> dict[str,int] | None:
    # try to make an mtime cache. If a file doesn't exist, return None
    out = {}
    for key, path in cache_keys.items():
        if not os.path.exists(path):
            return None
        out[key] = _get_mtime(path)
    return out

def _can_skip_doxygen_call(cache_file:str, cache_keys: dict[str, os.PathLike]) -> bool:
    # load the cached_mtimes
    try:
        with open(cache_file, "r") as f:
            cached_mtimes = json.load(f)
    except FileNotFoundError:
        return False  # if we can't find the cached mtimes, we need to call doxygen

    # measure the modification times
    actual_mtimes = _try_make_mtime_cache(cache_keys)

    if actual_mtimes is None:
        return False  # a file was missing, so we need to regenerate doxygen outputs

    return actual_mtimes == cached_mtimes


def build_doxygen(doxybuild_build_cache_dir: os.PathLike):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cur_dir, "..", "..", ".."))
    src_dir = os.path.join(root_dir, "src")
    doxygen_dir = os.path.join(root_dir, "docs", "doxygen")
    default_doxyfile = os.path.abspath(os.path.join(doxygen_dir, "Doxyfile"))

    custom_doxyfile = os.path.abspath(os.path.join(doxybuild_build_cache_dir, "Doxyfile"))

    template_mapping = {
        "RAW_CONF_FILE": default_doxyfile,
        "INPUT": os.path.abspath(src_dir),
        "OUTPUT_DIRECTORY":doxybuild_build_cache_dir
    }
    dox_html_outdir = os.path.join(doxybuild_build_cache_dir, "html")

    # to try to reduce unnecessary modification times, we'll check modification times
    # for each of the files/directories listed in the following directory
    cache_keys = {
        'src' : src_dir,
        'doxygen' : default_doxyfile,
        'generated_doxyfile' : custom_doxyfile,
        'dox-build-dir' : dox_html_outdir
    }

    # we will compare this against data previously saved at the path
    cache_file = os.path.join(doxybuild_build_cache_dir, "cached_mtimes.json")

    if _can_skip_doxygen_call(cache_file, cache_keys):
        return dox_html_outdir

    if os.path.exists(custom_doxyfile):
        os.remove(custom_doxyfile)

    if os.path.exists(dox_html_outdir):
        shutil.rmtree(dox_html_outdir)

    with open(custom_doxyfile, "w") as f:
        _write_template(f, template=_TEMPLATE, mapping=template_mapping)

    try:
        retcode = subprocess.call(["doxygen", custom_doxyfile])

        if retcode != 0:
            sys.stderr.write("doxygen terminated by signal %s" % (-retcode))
            if os.path.exists(dox_html_outdir):
                shutil.rmtree(dox_html_outdir)
        else:
            # get modification time of dox_html_outdir (after the build)
            mtime_pack = _try_make_mtime_cache(cache_keys)
            assert mtime_pack is not None # sanity check!
            with open(cache_file, "w") as f:
                json.dump(mtime_pack, f)
    except OSError as e:
        sys.stderr.write(f"doxygen execution failed: {e}")

    return dox_html_outdir


def generate_and_copy_doxygen(doxybuild_build_cache_dir, dest_dir):
    # a short-term hack to get the build-directory
    # -> the proper fix involves making a sphinx extension
    dox_html_outdir = build_doxygen(doxybuild_build_cache_dir)

    # copy the files into the builddir
    # TODO: build them at the destination to begin with
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copytree(src=dox_html_outdir, dst=dest_dir, dirs_exist_ok=True)


def setup(app):
    # just like breathe, we're going to assume that the general build directory is
    # the parent of the doctree directory
    build_dir = os.path.dirname(os.path.abspath(app.doctreedir))

    # we will use this directory as a caching location
    doxybuild_build_cache_dir = os.path.join(build_dir, "doxybuild-cache")

    if not os.path.exists(doxybuild_build_cache_dir):
        os.makedirs(doxybuild_build_cache_dir)

    if os.getenv("SKIPDOXYGEN", "FALSE").lower() == "true":
        pass  # do nothing!
    else:
        generate_and_copy_doxygen(
            doxybuild_build_cache_dir=doxybuild_build_cache_dir,
            dest_dir=os.path.join(app.outdir, "Reference", "internal-api-ref")
        )
