#!/usr/bin/env python3
"""
A script used in continuous integration to setup the cholla submodule.

This (hopefully) provides a more robust, repeatable experience.
"""  # this docstring is reused for argparse's description argument

# for portability: only use standard-library modules present in older python versions
import argparse
import logging
import os
import subprocess
import sys
import traceback
from typing import Dict, IO, Mapping, Optional

# Handle some global stuff
# ========================
if sys.version_info < (3, 6):
    raise RuntimeError("python 3.6 or newer is required")

logger = logging.getLogger("setup")
logger.setLevel(logging.DEBUG)


def _configure_logger(color=False):
    global logger
    fmt = "%(name)s > %(message)s"
    if color:
        fmt = "\x1b[36;20m" + fmt + "\x1b[0m"

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
        logger.info(f"{_msg}; ({'; '.join(_meta_list)})")

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


def _check_large_files_were_downloaded(submodule_path: str):
    import json
    import tempfile

    # query the list of all files tracked by git-lfs (in json format)
    with tempfile.TemporaryFile() as tmp_fp:
        _run(
            *["git", "lfs", "ls-files", "--json"],
            log=False,
            cwd=submodule_path,
            stdout=tmp_fp,
        )
        tmp_fp.seek(0)
        json_data = json.load(tmp_fp)

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
            path = os.path.join(submodule_path, finfo["name"])
            raise ScriptError(f"There seems to have been an issue downloading {path}")


def _setup_submodule(repo_path: Optional[str] = None):
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

    # next, we pull the git-lfs tracked data
    logger.info("Pre-fetch then Checkout data tracked by git-lfs")
    _run(
        *["git", "submodule", "foreach", "--recursive", "git", "lfs", "fetch"],
        cwd=repo_path,
    )
    _run(
        *["git", "submodule", "foreach", "--recursive", "git", "lfs", "checkout"],
        cwd=repo_path,
    )

    # perform a sanity check (maybe we should make this optional?)
    logger.info("Confirming that all large files were successfully downloaded")
    _check_large_files_were_downloaded(
        submodule_path=os.path.join(repo_path, "cholla-tests-data")
    )


def main(args: argparse.Namespace):
    _configure_logger(color=args.color)

    try:
        _setup_submodule(args.repo_path)

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
        traceback.print_exc(file=sys.stderr)
        return 70  # https://www.man7.org/linux/man-pages/man3/sysexits.h.3head.html
    else:
        logger.info("success")
        return 0


parser = argparse.ArgumentParser(
    description=__doc__.strip(),  # remove leading and trailing string newlines
    formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False,
)

parser.add_argument("--color", action="store_true", help="use color")
parser.add_argument(  # used for testing
    "--repo-path", default=None, help="optionally specify path to repository"
)

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))
