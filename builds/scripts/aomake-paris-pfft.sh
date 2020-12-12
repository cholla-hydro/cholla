#!/bin/bash

module use /home/users/twhite/share/modulefiles
module --no-pager load pfft-ompi
module load hdf5
module list

export CC=mpicc
export CXX=mpicxx
export HIP_PLATFORM=hcc
export OMP_NUM_THREADS=16
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd-ompi'
export TYPE=gravity

make clean
make -j
