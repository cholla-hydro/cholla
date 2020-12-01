#!/bin/bash

module use /home/users/twhite/share/modulefiles
module load ompi-cray hdf5
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CC=mpicc
export CXX=mpicxx
export HIP_PLATFORM=hcc
export OMP_NUM_THREADS=16
export POISSON_SOLVER='-DSOR -DPARIS'
export SUFFIX='.paris.sor.ompi'
export TYPE=gravity

make clean
make -j
