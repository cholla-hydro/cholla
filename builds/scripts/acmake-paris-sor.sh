#!/bin/bash

module restore -s PrgEnv-cray
module unload cray-mvapich2
module use /home/groups/coegroup/sabbott/cray-mpi/modulefiles
module load cray-mpich
module load hdf5/1.10.1
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CC=cc
export CXX=CC
export HIPCONFIG=$(hipconfig -C)
export OMP_NUM_THREADS=16
export POISSON_SOLVER='-DSOR -DPARIS'
export SUFFIX='.paris.sor.cray'
export TYPE=gravity

make clean
make -j
