#!/bin/bash

module load hdf5
module load cuda
module list

#Load ics for scaling test
mkdir data
cd data
wget https://www.dropbox.com/s/ean9331oqacemlq/ics_25Mpc_128_particles.h5?dl=0

export POISSON_SOLVER='-DPARIS'
export SUFFIX='.paris-cuda'
make clean
make -f Makefile_FOM.sh
