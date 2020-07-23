#!/bin/bash

if [ "$1" == "summit" ]; then
  if ! module is-loaded gcc hdf5 cuda; then
    echo "Please first load the modulefile for gcc, hdf5, and cuda"
    echo "e.g.: 'module load gcc hdf5 cuda'"
    exit 1
  fi
fi
