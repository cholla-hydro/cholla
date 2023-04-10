#!/usr/bin/env bash

# Description:
# Run clang-format on all the source files in Cholla. Any command line arguments
# provided to this script are passed directly to clang-format
#
# Dependencies:
# - clang-format v15 or greater
# - GNU Find, the default macos version won't work

# Get the location of Cholla
cholla_root=$(git rev-parse --show-toplevel)

# Get a list of all the files to format
readarray -t files <<<$(find ${cholla_root} -regex '.*\.\(h\|hpp\|c\|cpp\|cu\|cuh\)$' -print)

for VAR in $LIST
do
    echo "$VAR"
done

#clang-format -i --verbose "$@" -style="file" "${files[@]}"
/usr/lib/llvm-15/bin/clang-format -i --verbose "$@" -style="file" "${files[@]}"