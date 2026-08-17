#!/bin/bash

module purge
source /vast/parcc/sw/lmod/z/go.sh
module load arch/b200/26.1
module load gcc 
module load cuda
module load openmpi
module load hdf5
module load miniconda3