#!/bin/bash

module load hdf5
module load cuda
module list

export POISSON_SOLVER='-DPARIS'
export SUFFIX='.paris-cuda'
make clean
make -f Makefile_FOM.sh
