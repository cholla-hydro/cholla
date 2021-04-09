#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.xl.sh 

module use /sw/spock/spack-envs/views/modules
module load cray-python
module load rocm/4.1.0
module load craype-accel-amd-gfx908
module load cray-hdf5 cray-fftw

#export GPU_MPI="-DGPU_MPI"
export F_OFFLOAD="-fopenmp"

export CHOLLA_ENVSET=1
