#!/bin/bash

module load hdf5/1.10.6 cuda10.2/10.2 openmpi/4.0.1

export MACHINE=lux
export CHOLLA_ENVSET=1
export PARIS_MPI_GPU="-DPARIS_NO_GPU_MPI" 
