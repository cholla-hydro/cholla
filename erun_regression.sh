#!/bin/bash
#BSUB -P CSC380
#BSUB -W 0:05
#BSUB -nnodes 2
#BSUB -J cholla
#BSUB -o o.%J
#BSUB -q debug

module load gcc hdf5 cuda

OUTDIR="out.regression.${LSB_JOBID}"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
jsrun --smpiargs="-gpu" -n8 -a1 -c1 -g1 ../cholla ../regression1.txt |& tee tee
