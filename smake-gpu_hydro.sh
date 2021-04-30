#!/bin/bash

#module load xl cuda hdf5
module load gcc/10.2.0 cuda hdf5

make clean
make TYPE=gpu_hydro -j 8
