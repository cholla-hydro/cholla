#!/bin/bash

module purge
module load craype-x86-naples craype-network-infiniband 
module load shared slurm
module use /home/users/twhite/share/modulefiles
module load ompi/4.0.4-rocm-3.9 hdf5

export OMPI_CC=$(which clang)
export OMPI_CXX=$(which clang)

export CHOLLA_MACHINE=poplar.aomp
export PYTHON=${HOME}/.venv/bin/python
