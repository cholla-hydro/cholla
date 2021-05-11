#!/bin/bash

module purge
module load craype-x86-naples
module load craype-network-infiniband
module load slurm shared

module load PrgEnv-cray
module load craype-accel-amd-gfx908
module unload cray-mvapich2
module use /home/groups/coegroup/sabbott/cray-mpi/modulefiles
module load rocm/4.0.0
module load cray-mpich/rocm4.0
module load hdf5/1.10.1

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

#-- Enable  GPU-MPI, requires OpenMP offload
#export GPU_MPI="-DGPU_MPI"
export F_OFFLOAD="-fopenmp"

export PYTHON=${HOME}/.venv/bin/python
