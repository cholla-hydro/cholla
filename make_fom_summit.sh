#!/bin/bash

module load gcc hdf5 cuda
module list

#Load ics for scaling test
# mkdir data
# cd data
# wget https://www.dropbox.com/s/v5zzuk5ma1a3x6g/ics_25Mpc_128.h5
# wget https://www.dropbox.com/s/ean9331oqacemlq/ics_25Mpc_128_particles.h5
# wget https://www.dropbox.com/s/rbtoo3jx9a558ip/ics_25Mpc_256.h5
# wget https://www.dropbox.com/s/7bq5an37uudtlz0/ics_25Mpc_256_particles.h5
# cd ..
export SYSTEM_NAME="Summit"
export POISSON_SOLVER='-DPARIS'
export SUFFIX='.paris-cuda'
make clean
make -f Makefile_FOM.sh
