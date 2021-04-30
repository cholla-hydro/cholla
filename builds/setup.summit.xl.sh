#!/bin/bash
#module load xl cuda fftw hdf5 python
module load gcc/10.2.0 cuda hdf5

export GPU_MPI="-DGPU_MPI"
export F_OFFLOAD="-qsmp=omp -qoffload"
#export F_OFFLOAD='-foffload=nvptx-none="-lm -Ofast"'

export CHOLLA_ENVSET=1
