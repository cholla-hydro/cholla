#!/bin/bash

module load hdf5/1.10.6
module load openmpi/4.0.1-cuda
module load cuda10.1/10.1
module list

export MPI_HOME='/cm/shared/apps/openmpi/openmpi-4.0.1.cuda/'
export FFTW_ROOT='/data/groups/comp-astro/bruno/code_mpi_local/fftw-3.3.8'
export PFFT_ROOT='/data/groups/comp-astro/bruno/code_mpi_local/pfft'
export POISSON_SOLVER='-DPFFT'
export SUFFIX='.pfft'
make clean
make
