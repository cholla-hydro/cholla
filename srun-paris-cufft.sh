#!/bin/bash
#BSUB -P VEN114
#BSUB -W 0:05
#BSUB -nnodes 1
#BSUB -J cholla.paris.cufft
#BSUB -o o.%J
#BSUB -q debug

module load gcc hdf5 cuda

OUTDIR="out.paris.cufft.${LSB_JOBID}"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export OMP_NUM_THREADS=16
jsrun --smpiargs="-gpu" -n1 -a1 -c16 -g1 ../cholla.paris.cufft ../parameter_file.txt |& tee tee
