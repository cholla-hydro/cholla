#!/bin/bash

export MPI_HOME='/home/bruno/code/openmpi-4.0.4'
export POISSON_SOLVER='-DCUFFT'
export SUFFIX=''
echo $(POISSON_SOLVER)
# make clean
make