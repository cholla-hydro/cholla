#!/bin/bash

module load pfft-ompi hdf5
module list

export HIP_PLATFORM=hcc
export POISSON_SOLVER='-DCUFFT -DPARIS'
export SUFFIX='.paris.hipfft-ompi'
export CC=mpicc
export CXX=mpicxx

#make clean
make
