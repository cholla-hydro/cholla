#!/bin/bash

export MPI_HOME='/cm/shared/apps/openmpi/openmpi-4.0.1.cuda/'
export POISSON_SOLVER='-DSOR'
export SUFFIX='.sor'
# make clean
make