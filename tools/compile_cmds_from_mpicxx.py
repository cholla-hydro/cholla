#!/usr/bin/env python3
"""
A simple tool for inferring the compile-flags directly implicitly included by
an mpicxx compiler wrapper
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from typing import List, Mapping, Sequence, Optional, Union


def _try_cmd(
    cmd: Sequence[str], env: Mapping[str, str], *, return_code: int = 0
) -> Optional[str]:
    rslt = subprocess.run(cmd, env=env, capture_output=True, encoding="utf8")
    if rslt.returncode == 0:
        return rslt.stdout.strip()
    return None


# The logic in the following function (and JUST that function) was directly transcribed
# from the _MPI_interrogate_compiler function in the CMake file:
#   https://github.com/Kitware/CMake/blob/0fedf1592c23bb0386f50b4c4d7b2ade8f091a46/Modules/FindMPI.cmake
# Consequently, the logic in the function is licensed according to:
#
# Copyright 2000-2026 Kitware, Inc. and `Contributors <CONTRIBUTORS.rst>`_
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of Kitware, Inc. nor the names of Contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


def _infer_compile_flags(mpicxx_path: str, env: Mapping[str, str]) -> Optional[str]:
    """Infer the compile-flags associated with mpicxx_path"""
    env = dict(**env)

    # skip modifications to I_MPI_CXX and MPICH_CXX

    # Set these two variables for Intel MPI:
    #   - I_MPI_DEBUG_INFO_STRIP: It adds 'objcopy' lines to the compiler output. We
    #     support stripping them (see below), but if we can avoid them in the first
    #     place, we should.
    #   - I_MPI_FORT_BIND: By default Intel MPI makes the C/C++ compiler wrappers link
    #     Fortran bindings. This is so that mixed-language code doesn't require
    #     additional libraries when linking with mpicc. For our purposes, this makes
    #     little sense, since cholla only has no fortan
    for _env_var in ("I_MPI_DEBUG_INFO_STRIP", "I_MPI_FORT_BIND"):
        env[_env_var] = "disable"

    def _try_opt_pair(compile_opt, link_opt):
        compile_cmdline = _try_cmd([mpicxx_path] + compile_opt, env=env)
        if compile_cmdline is None:
            return (None, None)
        link_cmdline = _try_cmd([mpicxx_path] + link_opt, env=env)
        if link_cmdline is None:
            return (None, None)
        return (compile_cmdline, link_cmdline)

    compile_cmdline = None
    link_cmdline = None

    # Check whether the -showme:compile option works. This indicates that we have
    # either Open MPI or a newer version of LAM/MPI, and implies that -showme:link
    # will also work. Open MPI also supports -show, but separates linker and compiler
    # information
    compile_cmdline, link_cmdline = _try_opt_pair(["-showme:compile"], ["-showme:link"])

    # MPICH and MVAPICH offer -compile-info and -link-info.
    # For modern versions, both do the same as -show. However, for old versions, they
    # do differ when called for mpicxx and mpif90 and it's necessary to use them over
    # -show in order to find the removed MPI C++ bindings.
    if compile_cmdline is None:
        compile_cmdline, link_cmdline = _try_opt_pair(["-compile-info"], ["-link-info"])

    # Cray compiler wrappers come usually without a separate mpicc/c++/ftn, but offer
    # --cray-print-opts=...
    # -> when checking link opts, pass --no-as-needed so the mpi library is always
    #    linked. Otherwise, the Cray compiler wrapper puts an --as-needed flag around
    #    the mpi library, and it is not linked unless code directly refers to it.
    if compile_cmdline is None:
        compile_cmdline, link_cmdline = _try_opt_pair(
            ["--cray-print-opts=cflags"], ["--no-as-needed", "--cray-print-opts=libs"]
        )

    # MPICH, MVAPICH2 and Intel MPI just use "-show". Open MPI also offers this, but the
    # -showme commands are more specialized.
    if compile_cmdline is None:
        compile_cmdline = _try_cmd([mpicxx_path, "-show"], env=env)

    # Older versions of LAM/MPI have "-showme". Open MPI also supports this.
    # Unknown to MPICH, MVAPICH and Intel MPI.
    if compile_cmdline is None:
        compile_cmdline = _try_cmd([mpicxx_path, "-showme"], env=env)

    # at this point, let's just return the results
    # -> I think we hit the vast majority of cases
    # -> most (all?) of the remaining logic has to do with cleaning up or extracting
    #    information from the compilation commands (I don't think we need to worry
    #    about doing this if we're just going to use the information for creating a
    #    compile_commands.json file
    return compile_cmdline


def infer_compile_flags(
    mpicxx_cmd: str,
    *,
    split_args: bool = False,
    env: Optional[Mapping[str, str]] = None,
) -> Union[str, List[str], None]:
    """Infer the compile-flags associated with mpicxx_path.

    Returns None upon failure
    """
    if os.path.isfile(mpicxx_cmd):
        mpicxx_path = mpicxx_cmd
    else:
        mpicxx_path = shutil.which(mpicxx_cmd)
    env = os.environ if env is None else env

    compile_flags = _infer_compile_flags(mpicxx_path=mpicxx_path, env=env)
    if split_args and compile_flags is not None:
        return shlex.split(compile_flags)
    return compile_flags


def main(args: argparse.Namespace) -> int:
    result = infer_compile_flags(args.mpicxx_wrapper)
    if result is not None:
        print(result)
        return 0
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mpicxx_wrapper", help="the mpi wrapper to interrogate")
    sys.exit(main(parser.parse_args()))
