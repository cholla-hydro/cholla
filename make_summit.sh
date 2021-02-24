#!/bin/bash

module load gcc
module load hdf5
module load cuda

export MPI_HOME=$(MPI_ROOT)
export POISSON_SOLVER='-DPARIS'
export SUFFIX=''
make -f Makefile_cosmo.sh clean
make -f Makefile_cosmo.sh 