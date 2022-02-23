#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.gcc.sh

module load gcc/10.2.0 cuda/11.4.0 fftw hdf5 python

export F_OFFLOAD="-fopenmp -foffload=nvptx-none='-lm -Ofast'"
export CHOLLA_ENVSET=1
