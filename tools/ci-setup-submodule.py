#!/usr/bin/env python3
"""
This script is used in CI to setup Cholla's data-file submodule.

Motivation
==========
At the time of writing, Cholla uses a git-submodule to track data-files that
are used for running a subset of tests. Naturally, CI workflows that run
Cholla's tests must check out this submodule.

Because of the size of the files, GitHub required us to use git-lfs to track
the large files.

The primary reason for the creation of this script is that we have been
encountering some weird intermittent challenges with checking out the
submodule in our Jenkins Instance. This instance runs on the cluster
managed by the CRC at the University of Pittsburgh). It seems probable that
the underlying are tied to well-known latency issues related to the cluster's
distributed filesystem.

Context: How git-lfs works
==========================
`git` generically provides support for "smudge" filters to modify files as
they are checked out. It allows external programs (like git-lfs) to create new
kinds of filters as a generic way to extend Git's functionality.

`git-lfs` creates smudge-filters for tracking large files. Rather than
tracking large files directly as part of the repository, `git-lfs` instead
has users track tiny (<= 1 kB) "pointer-files" that contain the size and an
object-id of the large file.

> Aside: an object-id is a generic git concept - its a checksum based on file
> contents that is used internally as a unique identifier of the file.

`git-lfs` provides a smudge-filter that uses the contents of each
"pointer-file" to download and replace it with the corresponding large file.

Under normal operation (and when `git-lfs` is installed), the process of a
`git-checkout` will seamlessly trigger `git-lfs` behind the scenes to replace
all of the pointer files being checked out. (It will be relevant later that
we can temporarily disable `git-lfs`)

Setting up the Submodule
========================
Since we encounter problems on the CRC's file-system when we invoke the
following command

```sh
$ git submodule update --init --recursive
```

we instead adopt a more manual procedure that achieves equivalent results. The
steps to our procedure include:

1. "initialize" the submodule (`git submodule init`)

2. pull the submodule data (`git submodule update`).

   - Importantly, we use an environment variable to disable all "smudging"
     pertaining to `git-lfs` when `git submodule update` internally triggers
     machinery equivalent to `git-clone`, `git-fetch`, and `git-checkout`
     for each submodule.

   - Aside: While `git-lfs` provides a few ways to disable smudging, a lot of
     trial-and-error suggests that the environment variable seems to be the
     only way to do it in the context of git submodule.

3. pre-fetch all of the relevant git-lfs data

4. actually checkout the git-lfs data

This procedure still fails on the CRC cluster. At the time of writing, all
failures seem to occur in step 3 (with slightly more descriptive error
messages than the use of the single flag.

A Manual Fallback Strategy
==========================
A lot of machinery in this script exists to support a (crude) fallback
strategy to retrieve the test-data after git-lfs fails. The strategy involves

1. iterating over all pointer-files in the git-submodule

2. using information about the file path to construct urls, where you can
   directly download files from GitHub

3. downloading the file from the url and validating its checksum (the checksum
   is provided by the pointer-file)

4. replacing the pointer-file with the downloaded file

This definitely "gets the job done," there are some concerns:
- If we aren't careful, we could potentially hit GitHub's internal limits
  for these kinds of downloads (see https://stackoverflow.com/a/74960542)
- frankly, it doesn't like a great idea to directly replace a file in the git
  repository

(We can probably overcome both issues)
"""

# for portability: only use standard-library modules present in older python versions
import argparse
import hashlib
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, IO, Iterable, Mapping, NamedTuple, Optional

# Handle some global stuff
# ========================
if sys.version_info < (3, 6, 1):  # 3.6.0 doesn't support all NamedTuple features
    raise RuntimeError("python 3.6.1 or newer is required")

logger = logging.getLogger("setup")
logger.setLevel(logging.DEBUG)

_CHUNKSIZE = 8192  # default chunksize used for file operations


def _configure_logger(color=False):
    global logger

    color_start = ""
    color_stop = ""
    if color:
        color_start = "\x1b[36;20m"
        color_stop = "\x1b[0m"

    fmt = f"{color_start}%(name)s{color_stop} > %(message)s"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console_handler)


class ScriptError(RuntimeError):
    pass


def _fmt_env_args(
    include_outer_env: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Format a string representation conveying env variables as concisely as possible
    """
    # this assumes that the env-overwrites are short
    kv_pairs = [] if env is None else (f"{k}={v}" for k, v in env.items())
    if include_outer_env and env is None:
        return "<inherit>"
    elif include_outer_env:
        return f"<inherit>.update({'; '.join(kv_pairs)})"
    elif env is None:
        return "<no-env-vars>"
    else:
        return f"{{{'; '.join(kv_pairs)}}}"


def _get_subprocess_run_env_kwarg(
    include_outer_env: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, str]]:
    """Construct the env kwarg for subprocess.run"""
    if include_outer_env and env is None:
        return None  # subprocess simply inherits the environment variables
    elif include_outer_env:
        out = os.environ.copy()
        out.update(env)
        return out
    elif env is None:
        return {}  # subprocess is run with no environment variables
    else:
        return env


def _run(
    *args: str,
    log: bool = True,
    silent: bool = False,
    check_returncode: bool = True,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    include_outer_env: bool = True,
    env: Optional[Mapping[str, str]] = None,
    stdout: Optional[IO[str]] = None,
) -> int:
    """Invoke a command

    The interface is loosely inspired by the nox API

    Parameters
    ----------
    *args
        The command and its arguments
    log : bool
        When True, we log the command
    silent : bool
        When True, stdout and stderr are suppressed
    check_returncode : bool
        When True, we report an error for non-0 return codes
    cwd : str, optional
        Optionally specifies a directory to invoke the command from
    timeout : float, optional
        If the timeout expires, the subprocess will be killed and after it
        is done terminating, an exception is raised
    include_outer_env: bool = True,
        When True (the default), the subprocess inherits the environment of
        the current process
    env : dict, optional
        When specified, it's used to specify the subprocess's env variables.
        When include_outer_env is True, we overwrite variables.
    stdout
        Optionally specifies an open file object where to write the
        contents of stdout.

    Returns
    -------
    int
        The command's returncode
    """
    # some argument checking:
    if len(args) == 0:
        raise ValueError("args was not specified")
    elif not isinstance(args[0], str):
        raise TypeError(f"args[0], {args[0]!r}, isn't a str")

    if log:
        _msg = " ".join(args)
        _meta_list = []
        if cwd is not None:
            _meta_list.append(f"exec_dir: {cwd}")
        _env_str = _fmt_env_args(include_outer_env=include_outer_env, env=env)
        _meta_list.append(f"ENV: {_env_str}")
        logger.info(f"$ {_msg}; ({'; '.join(_meta_list)})")

    # adjust stdout if necessary
    if stdout is None and not silent:
        stdout = subprocess.PIPE
    elif stdout is not None and silent:
        raise ValueError("Can't specify stdout when silent=True")
    # define the value of stderr
    stderr = subprocess.STDOUT if silent else None

    rslt = subprocess.run(
        args,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        env=_get_subprocess_run_env_kwarg(include_outer_env=include_outer_env, env=env),
        timeout=timeout,
    )
    if check_returncode and (rslt.returncode != 0):
        if silent and rslt.stdout:
            print(rslt.stdout.decode("utf8"), file=sys.stderr, flush=True)
        cwd = "./" if cwd is None else cwd
        raise ScriptError(
            "subprocess exited with nonzero code\n"
            f"  command: {' '.join(args)}\n  exec_dir: {cwd!r}\n"
            f"  env: {_fmt_env_args(include_outer_env=include_outer_env, env=env)}\n"
            f"  code: {rslt.returncode}\n"
        )
    else:
        return rslt.returncode


# define the actual CI logic
# --------------------------
# -> the plan is to gradually script more and more CI log in python and move away
#    from shell-scripting. But, we are starting out with EXTREMELY simple logic

_keyvalue_regex = re.compile(r"(?P<key>[._a-z0-9]+) (?P<value>[^\r\n]+)\n")


class PointerFileInfo(NamedTuple):
    """
    Specifies information about a large file tracked by git-lfs
    """

    # path (on disk) to the repository holding the file
    full_file_path: str
    # path of the file relative to the root of the repository holding the file
    relative_to_repo_path: str
    # oid is the standard git abbreviation for object id. It is checksum that is
    # used to uniquely identify the file's contents
    oid: str
    # size specifies the full file's size
    size: int


def _parse_ptr_file(repo_location: str, relative_to_repo_path: str) -> PointerFileInfo:
    """
    Parse the contents of a git-lfs pointer file.

    Raises an exception if the file doesn't follow the specification:
        https://github.com/git-lfs/git-lfs/blob/8e6e9f1894d8ec89b74222c3fc00cb183959afd9/docs/spec.md
    """
    # to simplify code, we're slightly more permissive than the spec in 3 regards:
    # 1. technically, the spec states that pointer files don't exceed 1024 bytes
    #    (we allow 1025 bytes since there is some ambiguity about whether a
    #    trailing newline would count)
    # 2. we aren't strict about key ordering

    path = os.path.join(repo_location, relative_to_repo_path)

    def _mk_err(msg):
        return RuntimeError(f"`{path}` isn't a git-lfs pointer file: {msg}")

    with open(path, "rb") as f:
        # unsure if an extra trailing newline is allowed to be appended
        # (for safety, assume that it is allowed and doesn't affect size)
        buf = f.read(1026)
        if buf.endswith(b"\n\n"):
            buf = buf[:-1]
        if len(buf) > 1024:
            raise _mk_err("too large")
    try:
        contents = buf.decode(encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _mk_err("not utf-8 encoded") from None

    tmp = {}
    cur_pos = 0
    cur_line = 0
    for match in _keyvalue_regex.finditer(contents):
        if cur_pos != match.start():
            raise _mk_err(f"line {cur_line} isn't a standard key-value pair")
        tmp[match["key"]] = match["value"]
        cur_pos = match.end()
        cur_line += 1
    if cur_pos != len(contents):
        raise _mk_err(f"line {cur_line} isn't a standard key-value pair")
    return PointerFileInfo(
        full_file_path=path,
        relative_to_repo_path=relative_to_repo_path,
        oid=tmp["oid"],
        size=tmp["size"],
    )


def _iterate_over_ptr_file(submodule_path: str) -> Iterable[PointerFileInfo]:
    # iterate over all git-lfs pointer files

    def _run_gitlfs(*args) -> str:
        # a helper function that runs git-lfs and returns the output as a string
        with tempfile.TemporaryFile() as tmp_fp:
            full_command = ["git", "lfs"]
            full_command.extend(args)
            _run(*full_command, log=False, cwd=submodule_path, stdout=tmp_fp)
            tmp_fp.seek(0)
            return tmp_fp.read().decode("utf-8")

    # get the git-lfs version
    version_str = _run_gitlfs("--version")
    m = re.match(r"^git-lfs/(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version_str)
    if m is None:
        raise ScriptError(
            f"git-lfs returns a string with an unexpected format: {version_str!r}"
        )
    version = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    if version < (2, 6, 0):
        raise ScriptError(
            f"Skip ptr file iteration, since git-lfs version, {'.'.join(version)}, "
            "is older than 2.6.0"
        )
    elif version < (3, 7, 0):
        # elif True:
        # get a list of tracked file names (use -n flag rather than --name-only since
        # past release notes suggests that the latter was originally --names)
        list_string = _run_gitlfs("ls-files", "-n").rstrip()
        paths = list_string.splitlines()
        print("list of lfs-tracked files in submodule:")
        print(path)

        # test whether each path hold a "pointer file" or the file itself
        for p in paths:
            try:
                yield _parse_ptr_file(submodule_path, p)
            except RuntimeError:
                continue
    else:
        # This branch only works for newer versions of git-lfs. It only exists because
        # I didn't original;y realize that it depends on newer features (we keep it
        # because it is more efficient)
        import json

        json_string = _run_gitlfs("ls-files", "--json")
        json_data = json.loads(json_string)

        # some quick sanity checks on the format
        if "files" not in json_data:
            raise ScriptError(
                '"files" key is missing from json output of `git lfs ls-files` for '
                f"submodule @ {submodule_path}"
            )
        elif len(json_data) == 0:
            raise ScriptError(
                "there don't appear to be any files tracked by git-lfs in submodule @ "
                f"{submodule_path}"
            )
        elif ("name" not in json_data["files"][0]) or not isinstance(
            json_data["files"][0].get("checkout"), bool
        ):
            raise ScriptError(
                "Unexpected json format from `git lfs ls-files --json` (did the schema "
                "change between git-lfs versions)?"
            )

        # now confirm that each of the files was checked out
        for finfo in json_data["files"]:
            if not finfo["checkout"]:
                yield PointerFileInfo(
                    full_file_path=os.path.join(submodule_path, finfo["name"]),
                    relative_to_repo_path=finfo["name"],
                    oid=f"{finfo['oid_type']}:{finfo['oid']}",
                    size=finfo["size"],
                )


def _progress_bar(tot_bytes, silent=False):
    """provides a function for drawing/updating progress bars"""
    from math import log10

    ncols = shutil.get_terminal_size()[0] - 1
    power_div_3 = int(log10(tot_bytes) // 3) if tot_bytes > 0 else 0
    factor, unit = 1000.0**power_div_3, (" B", "KB", "MB", "GB")[power_div_3]
    # the output line has the form: '[<progress-bar>] <size>/<size> <unit>'
    fmt = "\r[{bar:{barlen}.{nfill}}] {size:.2f}" + f"/{tot_bytes / factor:.2f} {unit}"
    barlen = ncols - 19  # for context, 15 <= (len(fmt.format(...)) - barlen) <= 19
    suppress = (barlen < 1) or silent or not sys.stdout.isatty()
    bar = None if suppress else (barlen * "=")

    def _update(size):
        nonlocal bar
        if size is None and bar is not None:
            print(flush=True)
            bar = None
        elif bar is not None:
            nfill = int(barlen * (size / tot_bytes))
            val = fmt.format(bar=bar, barlen=barlen, nfill=nfill, size=size / factor)
            print(val, end="", flush=True)

    return _update


def _retrieve_url(
    url: str, dst: str, *, silent: bool = False, chunksize: int = _CHUNKSIZE
):
    """download the file from url to dst"""
    # delay the imports of the following modules since they are only used in
    # this function, and this function is only invoked as a fallback-plan
    import contextlib
    import urllib.request
    from urllib.error import URLError, HTTPError

    try:
        req = urllib.request.Request(url)
        with contextlib.ExitStack() as stack:
            out_file = stack.enter_context(open(dst, "wb"))
            response = stack.enter_context(urllib.request.urlopen(req))
            total_bytes = int(response.headers.get("Content-Length", -1))
            update_progress = _progress_bar(total_bytes, silent=silent)
            stack.callback(update_progress, size=None)

            # write downloaded data to a file
            downloaded_bytes = 0
            while True:
                update_progress(downloaded_bytes)
                block = response.read(chunksize)
                if not block:
                    break
                downloaded_bytes += len(block)
                out_file.write(block)
    except HTTPError as e:
        raise ScriptError(f"server can't fulfill request to fetch {url}: {e.code}")
    except URLError as e:
        raise ScriptError(f"server can't be reached to fetch {url}: {e.code}")


def calc_checksum(fname, alg_name, *, chunksize=_CHUNKSIZE):
    """Calculate the checksum for a given fname"""
    hash_calculator = hashlib.new(alg_name)
    with open(fname, "rb") as f:
        buffer = bytearray(chunksize)
        while True:
            nbytes = f.readinto(buffer)
            if nbytes == chunksize:
                hash_calculator.update(buffer)
            elif nbytes:  # equivalent to: (nbytes is not None) and (nbytes > 0)
                hash_calculator.update(buffer[:nbytes])
            else:
                break
    return ":".join([alg_name.lower(), hash_calculator.hexdigest()])


def _fallback_download(repo_path: Optional[str], relative_submodule_path: str):
    # this is an EXTREMELY hacky fallback scheme to try to manually download files if
    # git-lfs failed
    #
    # The docstring at the top of this file provides more context

    # get commit-hash associated with the submodule
    with tempfile.TemporaryFile() as tmp_fp:
        full_command = ["git", "rev-parse", f"HEAD:{relative_submodule_path}"]
        _run(*full_command, log=False, cwd=repo_path, stdout=tmp_fp)
        tmp_fp.seek(0)
        submodule_commit_hash = tmp_fp.read().decode("utf-8").rstrip()

    _run("git", "status", cwd = relative_submodule_path)
    startpath=relative_submodule_path
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')

    # maybe don't hardcode _URL_PREFIX in the future
    _URL_PREFIX = "https://github.com/cholla-hydro/cholla-tests-data"
    base_url = f"{_URL_PREFIX}/raw/{submodule_commit_hash}"

    # construct an iterator for all of the pointer files that were not checked out
    if repo_path is None:
        submodule_path = relative_submodule_path
    else:
        submodule_path = os.path.join(repo_path, relative_submodule_path)
    itr = _iterate_over_ptr_file(submodule_path)
    logger.info("HACKY WORKAROUND: attempt to manually fetch files tracked by git-lfs")
    logger.info(f"-> using <base-url>: {base_url}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dst_path = os.path.join(tmpdirname, "downloaded-file")
        for ptr_info in itr:
            logging_name = f"<base-url>/{ptr_info.relative_to_repo_path}"
            logger.info(f"try downloading {logging_name}")

            # download the file
            full_url = f"{base_url}/{ptr_info.relative_to_repo_path}"
            _retrieve_url(url=full_url, dst=tmp_dst_path)

            final_dst_path = ptr_info.full_file_path

            # validate checksum
            oid = ptr_info.oid.lower()
            alg_name, _ = ptr_info.oid.split(":")  # alg_name is sha256 (as of now)
            cksum = calc_checksum(tmp_dst_path, alg_name)
            if cksum != oid:
                raise ScriptError(f"downloaded {logging_name} has wrong checksum")

            # finally move the file to the final destination
            os.remove(final_dst_path)
            # we use shutil.move in case tmp_dst_path is on a different file-system
            shutil.move(tmp_dst_path, final_dst_path)


def _setup_submodule(
    repo_path: Optional[str] = None,
    simulate_lfs_fetch_failure: bool = False,
    fallback_manual_lfs_download: bool = False,
):
    """
    Encodes the actual logic for setting up the submodule

    For some context:
    - both git's submodule feature and the git-lfs are not particularly well
      regarded. My impression is that both of these things historically had
      major problems. While they both have come a long way and improved a lot,
      I think the sentiment remains that they are not very optimal tools
      (particularly git-lfs)
    - while both git's submodule feature and the git-lfs features do work
      together, there is a surprising lack of documentation about dealing with
      issues
    - things further get complicated while using them on shared file systems
      with high latencies (we have run into a bunch of intermittent issues
      with not downloading files tracked by git-lfs)
    """

    # The docstring at the top of this file provides more context

    # we currently assume that:
    # 1. the repository has already been cloned (it needs to be in order to be running
    #    this script)
    # 2. we are executing this script from the root of the repositry (we can adjust
    #    this assumption in the future)
    if repo_path is None:
        logger.info(f"Submodule Setup (assumed repo-path: {os.getcwd()})")
    else:
        logger.info(f"Submodule Setup (repo-path: {repo_path})")

    # first, off let's init the submodule
    logger.info("First, perform some basic submodule setup")
    _run("git", "submodule", "init", cwd=repo_path)

    # now, we fetch the submodule data without pulling data for the large files
    # tracked by git-lfs
    # -> instead we pull the pointer files (that instructs git-lfs where to get the
    #    data from)
    # -> I spent a lot of time trying to see if we could prefetch the git-lfs data, but
    #    that doesn't seem to be possible for git submodules
    # -> It appears that I NEED to use the environment variable to instruct git-lfs to
    #    not pull the big files. I also tried using
    #        `git lfs install --local --skip-smudge`
    #    but that doesn't work
    logger.info("Get the submodule data (without full data tracked by git-lfs)")
    _run(
        *["git", "submodule", "update", "--recursive"],
        cwd=repo_path,
        env={"GIT_LFS_SKIP_SMUDGE": "1"},
    )

    logger.info("Let's see if we can determine what wrong with the submodule")
    _run(
        *["git", "status"],
        cwd="./cholla-tests-data",
    )
    _run(
        *["git", "ls-files"],
        cwd="./cholla-tests-data",
    )
    _run(
        *["git", "fsck", "--full", "--verbose"],
        cwd="./cholla-tests-data",
    )

    logger.info("run some commands from root")

    _run(
        *["git", "ls-files"],
    )

    _run(
        *["git", "fsck", "--full"],
    )

    """
    # next, we pull the git-lfs tracked data
    logger.info("Pre-fetch then Checkout data tracked by git-lfs")
    try:
        if simulate_lfs_fetch_failure:  # for testing purposes
            logger.info("simulate failure of pre-fetch")
            raise ScriptError("simulated failure")
        _run(
            *["git", "submodule", "foreach", "--recursive", "git", "lfs", "fetch"],
            cwd=repo_path,
        )
        _run(
            *["git", "submodule", "foreach", "--recursive", "git", "lfs", "checkout"],
            cwd=repo_path,
        )
        lfs_success = True
    except ScriptError:
        lfs_success = False
        if not fallback_manual_lfs_download:
            raise

    if fallback_manual_lfs_download and not lfs_success:
        logger.info("Attempting to work around git-lfs failure")
        _fallback_download(
            repo_path=repo_path, relative_submodule_path="cholla-tests-data"
        )
    """

def main(args: argparse.Namespace):
    _configure_logger(color=args.color)

    if args.detailed_help:
        print(__doc__.strip())
        return 0

    try:
        _setup_submodule(
            repo_path=args.repo_path,
            simulate_lfs_fetch_failure=args.simulate_lfs_fetch_failure,
            fallback_manual_lfs_download=args.fallback_manual_lfs_download,
        )

    except ScriptError as err:
        # in this case, we handle "expected errors"
        # - these are things that *should* generally work, but could go wrong
        # - in general, these errors have nice error-messages and the standard python
        #   traceback would simply pollute this script's output
        # - an example is that a git command may fail because of a network issue or
        #   something unrelated to the core-logic in the script
        logger.error(f"ERROR: {err.args[0]}")
        return 70  # https://www.man7.org/linux/man-pages/man3/sysexits.h.3head.html
    except BaseException:
        # here we handle all other exceptions (e.g. programming errors,
        # KeyboardInterrupt). Generally we want a standard traceback in these cases
        logger.error("Unexpected error:")
        raise
    else:
        logger.info("success")
        return 0


parser = argparse.ArgumentParser(
    description=(
        "Helps setup the submodule for continuous integration. The --detailed-help "
        "flag will display an extended description of this tool's purpose and why "
        "it exists"
    ),
    allow_abbrev=False,
)

parser.add_argument("--color", action="store_true", help="use color")
parser.add_argument(
    "--detailed-help", action="store_true", help="shows the detailed help message"
)
parser.add_argument(  # used for testing
    "--repo-path", default=None, help="optionally specify path to repository"
)
parser.add_argument(  # used for testing
    "--simulate-lfs-fetch-failure",
    action="store_true",
    default=None,
    help=(
        "skip the git-lfs commands and act as if it failed. This is primarily intended "
        "for testing purposes"
    ),
)

parser.add_argument(
    "--fallback-manual-lfs-download",
    action="store_true",
    default=None,
    help=(
        "enables the fallback strategy when `git-lfs-fetch` or `git-lfs-checkout`fails"
    ),
)

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
