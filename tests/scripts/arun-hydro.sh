#!/bin/bash

module load rocm
module load PrgEnv-cray
module load hdf5
module load gcc/8.1.0

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
OUTDIR="run/out.hydro-amd.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=0
export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=16
srun -n1 -c$OMP_NUM_THREADS -N1 --exclusive -C MI60 ../../bin/cholla.hydro-amd ../../tests/3D/sod.txt |& tee tee.mi60
srun -n1 -c$OMP_NUM_THREADS -N1 --exclusive -p amdMI100 ../../bin/cholla.hydro-amd ../../tests/3D/sod.txt |& tee tee.mi100
