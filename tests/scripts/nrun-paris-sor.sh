#!/bin/bash

module restore PrgEnv-cray
module load hdf5
module load gcc/8.1.0

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
SUFFIX=paris.sor.cuda
OUTDIR="run/out.$SUFFIX.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=1
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=16
srun -n4 -c$OMP_NUM_THREADS -N1 --exclusive -p v100 ../../bin/cholla.$SUFFIX ../../tests/scripts/sphere.txt |& tee tee
