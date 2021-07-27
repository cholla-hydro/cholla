#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.summit.xl.sh 

module load cray-python
module load rocm/4.1.0
module load craype-accel-amd-gfx908
module load cray-hdf5 cray-fftw

#-- GPU-aware MPI
export PE_MPICH_GTL_DIR_amd_gfx908="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx908="-lmpi_gtl_hsa"
export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export MPI_GPU="-DMPI_GPU"
export F_OFFLOAD="-fopenmp"

export CHOLLA_ENVSET=1
