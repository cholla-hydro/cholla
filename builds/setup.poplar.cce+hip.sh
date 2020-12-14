#!/bin/bash

module purge
module load craype-x86-naples
module load craype-network-infiniband
module load slurm shared

module restore PrgEnv-cray
module load craype-accel-amd-gfx908
module unload cray-mvapich2
module load cray-mvapich2_nogpu
module load hdf5
module load rocm

export CHOLLA_MACHINE=poplar.cce+hip
export PYTHON=${HOME}/.venv/bin/python
