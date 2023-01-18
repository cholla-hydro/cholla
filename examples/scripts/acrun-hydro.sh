#!/bin/bash

module restore -s PrgEnv-cray
module unload cray-mvapich2
module use /home/groups/coegroup/sabbott/cray-mpi/modulefiles
module load cray-mpich
module load hdf5/1.10.1
module list

export LD_LIBRARY_PATH="$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH"
OUTDIR="run/out.hydro-amd-cray.$(date +%m%d.%H%M%S)"
set -x
mkdir -p ${OUTDIR}
cd ${OUTDIR}
export OMP_NUM_THREADS=16
srun --mpi=pmi2 -n1 -c$OMP_NUM_THREADS -N1 --exclusive -p amdMI100 ../../bin/cholla.hydro-amd-cray ../../examples/3D/sod.txt |& tee tee.mi100
