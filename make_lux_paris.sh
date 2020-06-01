#!/bin/bash

module load hdf5/1.10.6
module load openmpi/4.0.1-cuda
module load cuda10.1/10.1
module list

export MPI_HOME='/cm/shared/apps/openmpi/openmpi-4.0.1.cuda/'
export GRAKLE_HOME='/home/brvillas/code/grackle'
CC	= mpicc
CXX   = mpic++
export POISSON_SOLVER='-DPARIS'
export SUFFIX='.paris'
make clean
make
