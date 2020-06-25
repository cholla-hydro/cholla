#!/bin/bash

module use /home/users/twhite/share/modulefiles
module load ompi-cray hdf5
module list

export HIP_PLATFORM=hcc
export POISSON_SOLVER="-DPARIS"
export SUFFIX='.paris-amd-ompi'
export CC=mpicc
export CXX=mpicxx
make clean
make -j
