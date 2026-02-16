"""
Build doxygen documentation.
"""

from collections.abc import Mapping
from collections import ChainMap
import os
import shutil
import textwrap

from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

from .build_snap import build_consistent_with_cache, try_measure_snap
from .run_doxygen import DoxyBuildPaths, run_doxygen, write_custom_doxyfile_if_needed


def generate_doxygen(
    build_paths: DoxyBuildPaths,
    html_file_extension: str = ".html",
    extra_overrides: Mapping[str, str] | None = None,
):
    baseline = {"HTML_FILE_EXTENSION": html_file_extension}

    if extra_overrides is None:
        extra_overrides = baseline
    else:
        extra_overrides = ChainMap(extra_overrides, baseline)

    # write the generated doxygen config file to override certain properties
    # (overwrite any existing file if it doesn't match)
    reuse_custom_doxyfile = not write_custom_doxyfile_if_needed(
        build_paths, extra_overrides
    )

    cache_file = os.path.join(build_paths.build_cache_dir, "cached_mtimes.json")
    if reuse_custom_doxyfile and build_consistent_with_cache(cache_file, build_paths):
        pass  # we can skip the build!
    else:
        success = run_doxygen(build_paths)
        # write a snapshot on success
        if success:
            try_measure_snap(build_paths, loudly_fail=True).write_json(cache_file)

    # copy the files into the builddir
    # TODO: DO THIS AT THE VERY END OF A BUILD


def setup_stub_files(app: Sphinx) -> None:
    """
    Write a stub file into the source-directory location where we will be
    copying the doxygen-generated index file at the end of the build

    We need to do this so that Sphinx-Generated Table Of Contents link
    to the doxygen page properly links agains the doxygen webpages
    """
    dest_dir = os.path.join(app.srcdir, app.config.doxybuild_dest_dir)
    stub_file = os.path.join(dest_dir, "index.rst")
    if not os.path.exists(stub_file):
        if not os.path.isdir(dest_dir):
            os.mkdir(dest_dir)
        contents = textwrap.dedent("""\
            DUMMY_TITLE
            ===========
            I am a dummy file that was written by the doxybuild Sphinx Extension.

            My purpose is to get processed by Sphinx so that Sphinx generates a dummy
            webpage and properly links to that page.
            The doxybuild extension should then replace this page with the
            doxygen-generated webpages.

            Something went wrong if you are reading this on a rendered webpage!
            """)
        with open(stub_file, "w") as f:
            f.write(contents)


def copy_doxygen_html(app: Sphinx, exception: None) -> None:
    """
    Copy the previously generated doxygen html into the output directory
    of the sphinx build
    """
    dest_dir: os.PathLike = os.path.join(app.outdir, app.config.doxybuild_dest_dir)
    build_paths: DoxyBuildPaths = app.config.doxybuild_build_paths
    assert os.path.isdir(dest_dir)
    shutil.copytree(src=build_paths.dox_build_dir, dst=dest_dir, dirs_exist_ok=True)


_CONFIG_VALS = [  # fmt: (name, default, rebuild, types)
    # these first 2 paths are specified relative to the config directory
    ("doxybuild_hardcoded_doxyfile", None, "env", frozenset([str])),
    ("doxybuild_src_code_dir", None, "env", frozenset([str])),
    # specified relative to the config directory
    ("doxybuild_dest_dir", None, "env", frozenset([str])),
    # a dict holding override values
    ("doxybuild_overrides", None, "env", frozenset([dict])),
]


def setup(app: Sphinx) -> ExtensionMetadata:
    if os.getenv("SKIPDOXYGEN", "FALSE").lower() != "true":
        for name, default, rebuild, types in _CONFIG_VALS:
            app.add_config_value(name, default, rebuild, types=types)

        app.connect("builder-inited", setup_stub_files)
        app.connect("build-finished", copy_doxygen_html)

        app.config.doxybuild_build_paths = DoxyBuildPaths.create(
            # just like breathe, we're going to assume that the general build directory
            # is the parent of the doctree directory, and we'll create a cache location
            # location there for our own use
            doxybuild_build_cache_dir=os.path.join(
                os.path.dirname(os.path.abspath(app.doctreedir)), "doxybuild-cache"
            ),
            cpp_src_dir=os.path.join(app.confdir, app.config.doxybuild_src_code_dir),
            hardcoded_doxyfile=os.path.join(
                app.confdir, app.config.doxybuild_hardcoded_doxyfile
            ),
        )

        # STEP 2: actually run doxygen and generate files (these will be copied later)
        html_file_suffix = app.config.html_file_suffix
        html_file_suffix = ".html" if html_file_suffix is None else html_file_suffix
        generate_doxygen(
            build_paths=app.config.doxybuild_build_paths,
            html_file_extension=html_file_suffix,
            extra_overrides=app.config.doxybuild_overrides,
        )
    return {"version": "0.1", "parallel_read_safe": False, "parallel_write_safe": False}
