#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.gcc.sh 


#-- Using latest GCC in User Managed Software (UMS) and latest CUDA
#   on Summit
module load gcc/10.2.0 cuda/11.2.0 fftw hdf5 python

GCC_UMS_DIR=/sw/summit/ums/stf010/gcc
latest=$(ls --color=never ${GCC_UMS_DIR} | tail -n1)
export GCC_ROOT=$GCC_UMS_DIR/$latest

echo "Using GCC in $GCC_ROOT"

export PATH=$GCC_ROOT/bin:${PATH}

export OMPI_CC=${GCC_ROOT}/bin/gcc
export OMPI_CXX=${GCC_ROOT}/bin/g++
export OMPI_FC=${GCC_ROOT}/bin/gfortran
export LD_LIBRARY_PATH=${GCC_ROOT}/lib64:${LD_LIBRARY_PATH}

echo "mpicxx --version is: "
mpicxx --version

export GPU_MPI="-DGPU_MPI"
export F_OFFLOAD="-fopenmp -foffload=nvptx-none='-lm -Ofast'"

export CHOLLA_ENVSET=1
