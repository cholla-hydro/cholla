#!/bin/bash
module restore -s PrgEnv-cray
module load pfft
module load hdf5/1.10.1
module load gcc/8.1.0
module load rocm

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
OUTDIR="run/out.paris.pfft-amd.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export MV2_USE_CUDA=0
export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=16
srun -n4 -c$OMP_NUM_THREADS -N1 --exclusive -p amdMI100 ../../bin/cholla.paris.pfft-amd ../../examples/scripts/parameter_file.txt |& tee tee.mi100
