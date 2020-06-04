#!/bin/bash

module load rocm
module load PrgEnv-cray
module load hdf5
module load gcc
module list

export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export HIP_PLATFORM=hcc
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.hipfft'
export DFLAGS='-DPARIS_NO_GPU_MPI'
make clean
make
