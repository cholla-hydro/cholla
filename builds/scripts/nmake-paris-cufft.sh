#!/bin/bash
module restore PrgEnv-cray
module load hdf5
module load gcc/8.1.0
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CXX=CC
export HIP_PLATFORM=''
export MPI_HOME=$(dirname $(dirname $(which mpicc)))
export OMP_NUM_THREADS=16
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.cufft'
export TYPE=gravity

make clean
make -j
