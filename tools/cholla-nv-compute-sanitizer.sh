#!/usr/bin/env bash

# Utility script for running the NVIDIA Compute Sanitizer.
# The Compute Sanitizer provides 4 tool:
# - Memcheck: The memory access error and leak detection tool.
# - Racecheck: The shared memory data access hazard detection tool.
# - Initcheck: The uninitialized device global memory access detection tool.
# - Synccheck: The thread synchronization hazard detection tool.
#
# See the NVIDIA docs for more detail:
# https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html
#
# Syntax: compute-sanitizer [options] app_name [app_options]
#
# Compilation: Benefits from -G and -lineinfo. -Xcompiler -rdynamic for backtraces

# Memcheck args
# --leak-check full/no (default: no) full = info about memory leaks
# --padding NUM, puts padding around arrays to improve out-of-bounds checking.
# NUM is The size of the pad in bytes, we should probably pad at least a couple
# of doubles, say 8 so pad=8*8=64
#
# initcheck args
# --track-unused-memory yes/no (default: no) Check for unused memory allocations.
#
# Racecheck args
# - --print-level info


#set -x #echo all commands
while getopts "t:h" opt; do
    case $opt in
        t)  # Set the tool to use
            case ${OPTARG} in
                m)
                    tool="memcheck"
                    tool_args="--leak-check full --padding 64 --report-api-errors all"
                    ;;
                r)
                    tool="racecheck"
                    tool_args="--print-level info"
                    ;;
                i)
                    tool="initcheck"
                    tool_args="--track-unused-memory yes"
                    ;;
                s)
                    tool="synccheck"
                    tool_args=""
                    ;;
            esac
            ;;
        h)  # Print help
            echo -e "
While not required the following compile flags can help: -G for debug builds,
-lineinfo for performance builds (can't be used with -G) and -Xcompiler -rdynamic
is useful for backtraces in all builds.

Options:
-t m/r/i/s: Selects the tool to use.
    m: runs the memcheck tool
    r: runs the racecheck tool
    i: runs the initcheck tool
    s: runs the synccheck tool
-h: This dialogue"
            exit 0
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            exit 1
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            exit 1
            ;;
    esac
done

# Exit if no tool was selected
if [ -z "$tool" ]; then
        echo 'Missing tool argument' >&2
        exit 1
fi

# Get Paths
cholla_root="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cholla_exe=$(find "${cholla_root}" -name cholla.*)
cholla_parameter_file="${cholla_root}/examples/3D/sod.txt"
COMPUTE_SANITIZER=$(which compute-sanitizer)
sanitizer_log_file="${cholla_root}/bin/compute-sanitizer-${tool}.log"

# Echo Paths
echo -e "cholla_root           = ${cholla_root}"
echo -e "cholla_exe            = ${cholla_exe}"
echo -e "cholla_parameter_file = ${cholla_parameter_file}"
echo -e "COMPUTE_SANITIZER     = ${COMPUTE_SANITIZER}"
echo -e "sanitizer_log_file    = ${sanitizer_log_file}"
echo -e ""
echo -e "tool      = ${tool}"
echo -e "tool_args = ${tool_args}"

# Execute Sanitizer
COMMAND="${COMPUTE_SANITIZER} --log-file ${sanitizer_log_file} --tool ${tool} ${tool_args} ${cholla_exe} ${cholla_parameter_file}"
echo -e "Launch Command = ${COMMAND}"
$COMMAND