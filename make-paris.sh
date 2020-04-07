#!/bin/bash

module load PrgEnv-cray
module load hdf5
module load gcc
module list

export MPI_HOME=$(echo "${PE_CRAY_FIXED_PKGCONFIG_PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/lib/pkgconfig,,')
export CC=cc
export CXX=CC
export POISSON_SOLVER='-DPARIS'
make clean
make -j
mv cholla cholla.paris
