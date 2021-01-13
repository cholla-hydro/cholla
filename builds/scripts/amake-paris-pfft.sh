#!/bin/bash
module use /home/users/twhite/share/modulefiles
module load pfft
module load hdf5
module load gcc/8.1.0
module load rocm
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"

export CXX=CC
export DFLAGS='-DPARIS_NO_GPU_MPI'
export HIP_PLATFORM=hcc
export MPI_HOME=$(dirname $(dirname $(which mpicc)))
export OMP_NUM_THREADS=16
export POISSON_SOLVER="-DPFFT -DPARIS"
export SUFFIX='.paris.pfft-amd'
export TYPE=gravity

make clean
make -j
