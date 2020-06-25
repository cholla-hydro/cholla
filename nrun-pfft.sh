#!/bin/bash

module load pfft
module load hdf5
module load gcc

OUTDIR="out.pfft.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=0
srun -n4 -c16 -N1 --exclusive -p v100 ../cholla.pfft ../parameter_file.txt |& tee tee
