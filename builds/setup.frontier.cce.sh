#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.frontier.cce.sh

module load cray-python
module load rocm
module load craype-accel-amd-gfx90a
module load cray-hdf5 cray-fftw
module load googletest/1.14.0

#module load cpe/26.03
#module load rocm
#module load cray-python
#module load craype-accel-amd-gfx90a
#module load amd/7.0.2
#module load cray-hdf5/1.14.3.9
#module load craype-x86-turin
#module load cray-fftw/3.3.10.11


#-- GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

#export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}:${ROCM_PATH}/lib
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export CHOLLA_ENVSET=1
