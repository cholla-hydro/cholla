#!/bin/bash
module restore PrgEnv-cray
module load hdf5
module load gcc/8.1.0
module load rocm
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CXX=CC
export HIPCONFIG=$(hipconfig -C)
export MPI_HOME=$(dirname $(dirname $(which mpicc)))
export OMP_NUM_THREADS=16
export SUFFIX='.hydro-amd'
export TYPE=hydro

make clean
make -j
