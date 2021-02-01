#!/bin/bash

module purge
module load gcc cuda openmpi hdf5
module list

make clean
make TYPE=MS
