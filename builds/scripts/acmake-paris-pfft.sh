#!/bin/bash

module restore -s PrgEnv-cray
module unload cray-mvapich2
module use /home/groups/coegroup/sabbott/cray-mpi/modulefiles
module load cray-mpich
module load hdf5/1.10.1
module use /home/users/twhite/share/modulefiles
module load pfft-cray
module list

export CC=cc
export CXX=CC
export HIPCONFIG=$(hipconfig -C)
export OMP_NUM_THREADS=16
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd-cray'
export TYPE=gravity

make clean
make -j
