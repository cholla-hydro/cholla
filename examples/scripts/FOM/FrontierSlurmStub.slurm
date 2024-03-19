#!/bin/bash -l
#SBATCH -J Cholla_CAAR_Cosmo_512
#SBATCH -N 64
#SBATCH -t 30:00
#SBATCH -A __project_id__
#SBATCH -o Cholla_CAAR_Cosmo_512.o%j

#-- set CHOLLA location
CHOLLA_HOME=../../../

#-- Set FOM data
FOM_DATA=/lustre/orion/csc380/proj-shared/cholla/FOM

ln -s ${FOM_DATA}/data 
ln -s ${FOM_DATA}/scale_output_files

EXECUTABLE=${CHOLLA_HOME}/bin/cholla.FOM.frontier

source ${CHOLLA_HOME}/builds/setup.frontier.cce.sh

export MPICH_ALLTOALL_SYNC_FREQ=2
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"
export MPICH_OFI_CXI_COUNTER_REPORT=2
export OMP_NUM_THREADS=7
export ROCFFT_RTC_CACHE_PATH=/dev/null

env &> job.environ
scontrol show hostnames > job.nodes
ldd $EXECUTABLE > job.exec.ldd

srun -u -N 64 -n 512  -c 7 --gpu-bind=closest --gpus-per-task=1 \
$EXECUTABLE ${CHOLLA_HOME}/examples/3D/CAAR_FOM_Cosmo_512.txt |& tee STDOUT
