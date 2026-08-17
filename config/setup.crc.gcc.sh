#!/bin/bash

#-- This script needs to be sourced in the terminal, e.g.
#   source ./setup.crc.gcc.sh

# module load python/anaconda3-2020.11 gcc/10.1.0 cuda/11.1.0 openmpi/4.0.5 hdf5/1.12.0 googletest/1.11.0
module load cuda/11.8 gcc/11.5.0-dak7qob openmpi/4.1.7-cuda hdf5/1.14.6 # for a100 nodes

echo "mpicxx --version is: "
mpicxx --version

# export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp"
export CHOLLA_ENVSET=1
