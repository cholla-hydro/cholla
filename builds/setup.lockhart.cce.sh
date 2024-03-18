#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.lockhart.cce.sh

module load cray-python
module load rocm
module load craype-accel-amd-gfx90a
module load cray-hdf5 cray-fftw

#-- GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

export CHOLLA_ENVSET=1
