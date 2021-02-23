#!/bin/bash

module load hdf5/1.10.4
module load cuda/10.1.243

export MPI_HOME=$(MPI_ROOT)
export POISSON_SOLVER='-DPARIS'
export SUFFIX=''
make -f Makefile_cosmo.sh clean
make -f Makefile_cosmo.sh 