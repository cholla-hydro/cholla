#!/bin/bash

module load PrgEnv-cray
module load hdf5
module load gcc
module list

export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export CC=cc
export CXX=CC
export POISSON_SOLVER=-DCUFFT
export SUFFIX='.cufft'
make clean
make -j
