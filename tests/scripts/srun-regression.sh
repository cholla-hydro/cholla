#!/bin/bash
#BSUB -P CSC380
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -J cholla
#BSUB -o o.%J
#BSUB -q debug

module load gcc hdf5 cuda

OUTDIR="run/out.regression.${LSB_JOBID}"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
jsrun --smpiargs="-gpu" -n1 -a1 -c1 -g1 ../../bin/cholla ../../tests/regression/hydro_input.txt |& tee tee
