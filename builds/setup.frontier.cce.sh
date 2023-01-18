#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.frontier.cce.sh

module load cray-python
module load rocm
module load craype-accel-amd-gfx90a
module load cray-hdf5 cray-fftw
module load googletest/1.10.0

#-- GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export CHOLLA_ENVSET=1
