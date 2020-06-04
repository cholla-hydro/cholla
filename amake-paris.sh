#!/bin/bash

module load rocm
module load PrgEnv-cray
module load hdf5
module load gcc
module list

export HIP_PLATFORM=hcc
export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export POISSON_SOLVER="-DPARIS"
export SUFFIX='.paris-amd'
make clean
make
