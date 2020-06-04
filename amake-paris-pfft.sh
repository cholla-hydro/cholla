#!/bin/bash

module load rocm
module load pfft
module load hdf5
module load gcc
module list

export HIP_PLATFORM=hcc
export MPI_HOME=$(echo "${PATH}" | sed 's,.*:\([^:]*\)mvapich\([^:]*\).*,\1mvapich\2,;s,/bin,,')
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd'
make clean
make
