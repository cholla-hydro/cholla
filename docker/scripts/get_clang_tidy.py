#!/usr/bin/env python3
"""
Fetch, build, and install a particular version of clang-tidy.

This is intended to be used within a Dockerfile. This is useful for both:
1. running clang-tidy directly via GitHub Actions (i.e. not just in Jenkins)
2. for devcontainers

This was heavily influenced by the shell scripts provided by LLVM at
- llvm/utils/docker/build_docker_image.sh
- llvm/utils/docker/scripts/checkout.sh
- llvm/utils/docker/scripts/build_install_llvm.sh
Both of these files are licensed under Apache-2.0 WITH LLVM-exception

In the future, we may want to add support for downloading from prebuilt
repositories for debian/ubuntu that are maintained by LLVM.
- However, doing that limits our control over specifying the precise
  minor/patch version of clang-tidy. Based on some recent reproducibility
  challenges, its a little unclear whether this is a significant issue.
- If we do want to take this approach, we could reuse logic from a script
  that I wrote for Grackle for this precise purpose
  https://github.com/grackle-project/grackle/blob/newchem-cpp/scripts/ci/install_dependencies.py
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger("get_clang_tidy")
logger.setLevel(logging.DEBUG)


def configure_logger(color: bool = False):
    """Configure logger outputs. Should be called once (& only once) @ startup."""
    global logger
    color_start, color_stop = ("\x1b[36;20m", "\x1b[0m") if color else ("", "")
    fmt = f"{color_start}%(name)s{color_stop} > %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)


def _run(
    *args: str, dry_run: bool = False, check: bool = True, **kw: Any
) -> subprocess.CompletedProcess:
    """Log and execute the specified command."""
    msg = f"$ {' '.join(args)}"
    if len(kw) != 0:
        msg += f"; {kw!r}"
    logger.info(msg)

    if dry_run:
        return subprocess.run("true", check=check)
    else:
        return subprocess.run(args, check=check, **kw)


def main(args: argparse.Namespace):
    configure_logger(color=args.color)

    target_llvm_version = args.llvm_version
    working_dir = "/tmp/clang-build"
    checkout_dir = os.path.join(working_dir, "src")
    build_dir = os.path.join(working_dir, "build")
    install_dir = "/tmp/clang-install"

    # clone llvm
    major_llvm_version = target_llvm_version.split(".")[0]
    git_cmd = [
        "git",
        "clone",
        f"--branch=release/{major_llvm_version}.x",
        "https://github.com/llvm/llvm-project.git",
        checkout_dir,
    ]
    _run(*git_cmd, dry_run=args.dry_run)

    # checkout appropriate branch
    tag = f"llvmorg-{target_llvm_version}"
    _run("git", "-C", checkout_dir, "checkout", tag, dry_run=args.dry_run)

    # move onto the build
    logger.info(f"ensure that {install_dir} exist")
    if not args.dry_run:
        os.makedirs(install_dir, exist_ok=True)

    # configure the build
    config_cmd = [
        "cmake",
        "-GNinja",
        "-DLLVM_ENABLE_PROJECTS=clang;clang-tools-extra",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DLLVM_ENABLE_ZSTD=OFF",  # <- inspired by clang-tidy-wheel
        f"-B{build_dir}",
        f"-S{os.path.join(checkout_dir, 'llvm')}",
    ]
    _run(*config_cmd, dry_run=args.dry_run)

    # run the build and install
    _install_targets = ["install", "install-clang-resource-headers"]
    _run("ninja", f"-C{build_dir}", *_install_targets, dry_run=args.dry_run)

    # cleanup from the build
    logger.info("clean up the build-dir")
    if not args.dry_run:
        shutil.rmtree(build_dir)
    return 0


_SHORT_DESCR = "Install clang-tidy"
parser = argparse.ArgumentParser(description=_SHORT_DESCR, allow_abbrev=False)
parser.add_argument("--color", action="store_true", help="use color")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument(
    "--work-dir", required=True, help="location where source code is cloned & built"
)
parser.add_argument(
    "--install-dir", required=True, help="location where build-products are installed"
)
parser.add_argument(
    "--llvm-version", required=True, help="specify desired clang-tidy version"
)

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
