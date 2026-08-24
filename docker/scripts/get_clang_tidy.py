#!/usr/bin/env python3
"""
Get a particular version of clang-tidy.

This tool is intended to help setup a Dockerfile. This is useful for both (i) running
clang-tidy directly via GitHub Actions and (ii) for devcontainers

The tool operates in 2 modes:

1. It can download a prebuilt version of clang-tidy (from the apt-repositories
   maintained by the LLVM project):
   - the user is only able to specify a major version of clang-tidy and then the
     newest available minor/patch version will be installed.
   - Be aware that the available clang-tidy versions are affected by the version of
     the operating system in the docker image.
   - this mode is a LOT faster and more robust
2. It performs a "full-build" of clang-tidy from the source code
   - the user specifies a particular major.minor.patch.
   - this approach is far less robust (it's not fully clear if it even works).
     Currently it's primarily being retained for historical purposes.
   - this logic was heavily influenced by the shell scripts provided by LLVM at
     - llvm/utils/docker/build_docker_image.sh
     - llvm/utils/docker/scripts/checkout.sh
     - llvm/utils/docker/scripts/build_install_llvm.sh
     Each of these files are licensed under Apache-2.0 WITH LLVM-exception
"""

import argparse
import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Iterator

if sys.version_info < (3, 10):

    def _os_release_lines() -> Iterator[str]:
        for path in ("/etc/os-release", "/usr/lib/os-release"):
            with contextlib.suppress(FileNotFoundError):
                with open(path, "r") as f:
                    yield from f
                return  # <- exit immediately
        raise OSError("unable to find '/etc/os-release' or '/usr/lib/os-release'")

    def freedesktop_os_release() -> Dict[str, str]:
        # crude backport of platform.freedesktop_os_release (which is based on
        # https://www.freedesktop.org/software/systemd/man/latest/os-release.html)

        escape_characters = [re.escape("\\"), "'", '"', "`", re.escape("$")]
        escape_pattern = re.escape("\\") + "[" + "".join(escape_characters) + "]"

        def sanitize_val_str(val):
            if (val[0] == val[-1]) and val[0] in ("'", '"'):
                val = val[1:-1]
            return re.sub(escape_pattern, lambda m: m.group(0)[1], val)

        out = {}
        for line in _os_release_lines():
            match = re.match(r"^([a-zA-Z0-9_]+)=(.*)$", line.rstrip())
            if match is not None:
                out[match.group(1)] = sanitize_val_str(match.group(2))
        return out

else:
    from platform import freedesktop_os_release


class LinuxDistroInfo:
    """Summarizes relevant information about linux distrbution"""

    def __init__(self):  # fails for non-linux systems or old/weird linux distributions
        data = freedesktop_os_release()
        self.id = data["ID"]  # <- standard guarantees its always defined (& lowercase)
        self.codename = data.get("CODENAME", None)
        if self.codename is None and self.id == "ubuntu":
            self.codename = data.get("UBUNTU_CODENAME", None)
        # self.version_id = data.get("VERSION_ID", None)


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


def _install_via_apt(pkg_names: List[str], dry_run: bool = False):
    _run("sudo", "apt-get", "-y", "update", dry_run=dry_run)
    _cmd = ["sudo", "apt-get", "install", "-y", "--no-install-recommends"] + pkg_names
    _run(*_cmd, dry_run=dry_run)


def setup_apt_llvm_repository(llvm_version: int, *, dry_run: bool = False):
    """This function registers LLVM's official apt repository"""
    # this logic is loosely based on the from https://apt.llvm.org/llvm.sh

    # infer linux distro information
    distro_info = LinuxDistroInfo()

    _descr = "to register LLVM's official apt repository"
    if distro_info.id not in ["ubuntu", "debian"]:
        raise RuntimeError(f"{_descr}: only possible on debian/ubuntu")
    elif distro_info.codename is None:
        raise RuntimeError(f"{_descr}: requires knowledge of the codename")

    logger.info("Fetch and register key for authenticating LLVM's APT repository")
    gpg_dst = "/etc/apt/trusted.gpg.d/apt.llvm.org.asc"
    gpg_src_url = "https://apt.llvm.org/llvm-snapshot.gpg.key"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, os.path.basename(gpg_dst))
        _run("wget", "-O", tmp_path, gpg_src_url, dry_run=dry_run)
        _run("sudo", "cp", tmp_path, gpg_dst, dry_run=dry_run)
        _run("sudo", "chmod", "a+r", gpg_dst, dry_run=dry_run)

    # Part 2: record the actual source of the package
    # -> the formula for determining the uri and suite-name is fairly
    #    consistent unless you are describing Debian Testing or the
    #    LLVM version under active development

    logger.info("Register the appropriate LLVM repository with APT")
    uri = f"http://apt.llvm.org/{distro_info.codename}/"
    suite = f"llvm-toolchain-{distro_info.codename}-{llvm_version}"
    cmd = (  # the last arg is intentionally a single string
        "add-apt-repository",
        "--yes",
        "--sourceslist",
        f"deb {uri} {suite} main",
    )
    _run("sudo", *cmd, dry_run=dry_run)


def _main_prebuilt(args: argparse.Namespace):
    """Install prebuilt clang-tidy from the apt-repositories maintained by the
    LLVM project."""

    version = args.llvm_version
    _version_err = False
    try:
        _version_number = int(version)
        _version_err = str(_version_number) != version
    except ValueError:
        _version_err = True
    if _version_err:
        raise RuntimeError(f"{args.llvm_version!r} isn't a major version number")
    dry_run = args.dry_run

    # setup the apt repository
    setup_apt_llvm_repository(llvm_version=version, dry_run=dry_run)

    # actually install the package
    # -> it is imperative that we install the libomp-<VERSION>-dev package so that
    #    clang-tidy is able to locate "omp.h" headers
    _install_via_apt(
        pkg_names=[f"clang-tidy-{version}", f"libomp-{version}-dev"], dry_run=dry_run
    )

    # make symlinks
    pairs = [
        (f"clang-tidy-{version}", "clang-tidy"),
        (f"run-clang-tidy-{version}.py", "run-clang-tidy.py"),
        (f"run-clang-tidy-{version}", "run-clang-tidy"),
    ]
    for pair in pairs:
        target, link_name = [os.path.join("/usr/bin", e) for e in pair]
        if not os.path.isfile(target):
            raise RuntimeError(f"{target} doesn't appear to exists")
        logger.info(f"make symlink: {link_name} -> {target}")
        if not dry_run:
            os.symlink(target, link_name)


def _main_full_build(args: argparse.Namespace):
    """Drive a full build of clang-tidy."""

    target_llvm_version = args.llvm_version
    working_dir = args.work_dir
    checkout_dir = os.path.join(working_dir, "src")
    build_dir = os.path.join(working_dir, "build")
    install_dir = args.install_dir

    # ensure that build-time dependencies are installed
    # -> the precise dependencies may depend on the ubuntu/debian version and on the
    #    the target llvm version, but that's ok for now
    _install_via_apt(
        pkg_names=[
            "ca-certificates",
            "cmake",
            "2to3",
            "python-is-python3",
            "subversion",
            "ninja-build",
            "git",
        ],
        dry_run=args.dry_run,
    )
    # shell-versions of this logic would subsequently run `rm -rf /var/lib/apt/lists/*`

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
    "--llvm-version", required=True, help="specify desired clang-tidy version"
)
subparsers = parser.add_subparsers(required=True)

prebuilt_parser = subparsers.add_parser(
    "prebuilt",
    description=(
        "Install prebuilt clang-tidy from apt-repositories maintained by LLVM project."
        "The --llvm-version flag should just be passed a major version number."
    ),
)
prebuilt_parser.set_defaults(fn=_main_prebuilt)

full_build_parser = subparsers.add_parser(
    "build-from-src",
    description=(
        "Attempt to build clang-tidy from source. The --llvm-version flag should be "
        "passed a major.minor.patch version number."
    ),
)
full_build_parser.add_argument(
    "--work-dir", required=True, help="location where source code is cloned & built"
)
full_build_parser.add_argument(
    "--install-dir", required=True, help="location where build-products are installed"
)
full_build_parser.set_defaults(fn=_main_full_build)


def main(args: argparse.Namespace):
    configure_logger(color=args.color)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
