#!/bin/bash

module use /home/users/twhite/share/modulefiles
module load pfft-ompi
module load hdf5
module list

export HIP_PLATFORM=hcc
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd-ompi'
export CC=mpicc
export CXX=mpicxx

make clean
make -j
