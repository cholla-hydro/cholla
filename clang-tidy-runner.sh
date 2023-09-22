#!/usr/bin/env bash

# Description:
# Run clang-tidy on all build types in parallel. Note that this spawns 2x the
# number of build types threads since each type has a thread for the CPU code
# and a thread for the GPU code

# If ctrl-c is sent trap it and kill all clang-tidy processes
trap "kill -- -$$" EXIT

# cd into the Cholla directory. Default to ${HOME}/Code/cholla
cholla_path=${1:-${HOME}/Code/cholla}
cd ${cholla_path}

# Run all clang-tidy build types in parallel
builds=( hydro gravity disk particles cosmology mhd dust)
for build in "${builds[@]}"
do
  make tidy TYPE=$build &
done

# Wait for clang-tidy to finish
wait
