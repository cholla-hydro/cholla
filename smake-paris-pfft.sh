#!/bin/bash

module load gcc hdf5 fftw cuda

export PFFT_ROOT=$(readlink -f ../share/pfft*) 
export MPI_HOME=${MPI_ROOT}
export HDF5INCLUDE=${OLCF_HDF5_ROOT}/include
export HDF5DIR=${OLCF_HDF5_ROOT}/lib
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft'
export CC=mpicc
export CXX=mpicxx
export LIBS="-L${CUDA_DIR}/lib64"
make clean
make
