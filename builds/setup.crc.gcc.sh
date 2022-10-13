#!/bin/bash

#-- This script needs to be sourced in the terminal, e.g.
#   source ./setup.crc.gcc.sh

module load python/anaconda3-2020.11 gcc/10.1.0 cuda/11.1.0 openmpi/4.0.5 hdf5/1.12.0 googletest/1.11.0

echo "mpicxx --version is: "
mpicxx --version

# export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp -foffload=disable"
export CHOLLA_ENVSET=1
