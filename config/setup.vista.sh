#!/bin/bash

#-- This script needs to be source-d in the terminal, e.g.
#   source ./setup.vista.gcc.sh

module load cuda/12.5 openmpi/5.0.5 hdf5/1.14.4 nvidia_math/12.4

export CHOLLA_ENVSET=1
