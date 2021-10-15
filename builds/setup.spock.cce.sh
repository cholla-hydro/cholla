#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.xl.sh

module load cray-python
module load rocm
module load craype-accel-amd-gfx908
module load cray-hdf5 cray-fftw

#-- GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp"

export CHOLLA_ENVSET=1


