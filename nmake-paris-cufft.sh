#!/bin/bash

module load PrgEnv-cray
module load hdf5
module load gcc
module list

export CXX=CC
export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.cufft'
make clean
make -j
