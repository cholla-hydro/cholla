#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.elkhorn.cce.sh

#module load gnu15 openmpi5 hdf5 nvidia-hpc-sdk cuda/13.1.1
#module load ohpc gnu15 git
module load ohpc gnu14 git
module load openmpi5 hdf5 nvidia-hpc-sdk cuda/12.8.1

#-- GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

export CHOLLA_ENVSET=1
