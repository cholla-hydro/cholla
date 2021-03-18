#!/bin/bash

module purge
module load gcc/10.1.0
module load cuda/11.1.0
module load openmpi/4.0.5
module load hdf5/1.12.0
module list

export POISSON_SOLVER='-DSOR'
export SUFFIX='.sor'
make clean
make TYPE=gravity 
