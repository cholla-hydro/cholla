#!/bin/bash

module load gcc hdf5 cuda
module list

export HDF5INCLUDE=${OLCF_HDF5_ROOT}/include
export HDF5DIR=${OLCF_HDF5_ROOT}/lib
export MPI_HOME=${MPI_ROOT}
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.cufft'
export CC=mpicc
export CXX=mpicxx
export LIBS="-L${CUDA_DIR}/lib64"
make clean
make TYPE=gravity -j
