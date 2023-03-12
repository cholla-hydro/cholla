#!/bin/bash

#-- This script needs to be sourced in the terminal, e.g.
#   source ./setup.c3po.gcc.sh

echo "mpicxx --version is: "
mpicxx --version

# export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp"
export CHOLLA_ENVSET=1
