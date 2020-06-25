#!/bin/bash

module load pfft-ompi hdf5

OUTDIR="out.paris.pfft-amd-ompi.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
srun -n4 -c16 -N1 --exclusive -p amdMI60 ../cholla.paris.pfft-amd-ompi ../parameter_file.txt |& tee tee
