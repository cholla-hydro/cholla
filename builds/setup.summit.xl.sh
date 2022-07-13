#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.xl.sh

module load xl cuda fftw hdf5 python googletest/1.11.0

export F_OFFLOAD="-qsmp=omp -qoffload"
export CHOLLA_ENVSET=1
