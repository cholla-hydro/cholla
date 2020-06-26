#!/bin/bash

module use /home/users/twhite/share/modulefiles
module load ompi-cray hdf5

OUTDIR="out.paris.hipfft-ompi.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=0
export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
export MV2_ENABLE_AFFINITY=0
srun -n1 -c16 -N1 --exclusive -p amdMI60 ../cholla.paris.hipfft-ompi ../parameter_file.txt |& tee tee
