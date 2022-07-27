#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.gcc.sh

#module load gcc/10.2.0 cuda/11.4.0 fftw hdf5 python
module load gcc cuda fftw hdf5 python googletest/1.11.0

#export F_OFFLOAD="-fopenmp -foffload=nvptx-none='-lm -Ofast'"
export F_OFFLOAD="-fopenmp -foffload=disable"
export CHOLLA_ENVSET=1
