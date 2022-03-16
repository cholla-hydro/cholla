#!/bin/bash

module restore PrgEnv-cray
module load hdf5
module load gcc/8.1.0
module list

export CXX=CC
export MPI_HOME=$(dirname $(dirname $(which mpicc)))
export OMP_NUM_THREADS=16
export POISSON_SOLVER="-DSOR"
export SUFFIX='.sor-cuda'
export TYPE=gravity

make clean
make -j
