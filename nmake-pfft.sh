#!/bin/bash

module load pfft
module load hdf5
module load gcc
module list

export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export CC=cc
export CXX=CC
export POISSON_SOLVER=-DPFFT
export SUFFIX='.pfft'
make clean
make -j
