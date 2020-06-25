#!/bin/bash

module load rocm
module load pfft
module load hdf5
module load gcc
module list

export CXX=CC
export HIP_PLATFORM=hcc
export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd'
export DFLAGS='-DPARIS_NO_GPU_MPI'
make clean
make -j
