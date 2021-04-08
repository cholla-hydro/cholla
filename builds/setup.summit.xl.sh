#!/bin/bash
module load xl cuda fftw hdf5 python

export F_OFFLOAD="-qsmp=omp -qoffload"
export CHOLLA_ENVSET=1
