#!/bin/bash

export MPI_HOME='/home/bruno/code/openmpi-4.0.4'
export POISSON_SOLVER='-DPARIS'
export SUFFIX=''
make -f Makefile_cosmo.sh clean
make -f Makefile_cosmo.sh