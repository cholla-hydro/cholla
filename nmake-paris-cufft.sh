#!/bin/bash

module load PrgEnv-cray
module load hdf5
module load gcc
module list

export MPI_HOME=$(echo "${PE_CRAY_FIXED_PKGCONFIG_PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/lib/pkgconfig,,')
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.cufft'
make clean
make
