#!/bin/bash

if [ "$1" == "build" ]; then
  
  case $2 in
    summit)
      if ! module is-loaded gcc hdf5 cuda fftw; then
        echo "modulefile required: gcc, hdf5, fftw, and cuda"
        echo "do: 'module load gcc hdf5 cuda fftw'"
        exit 1
      fi
      ;;
    poplar)
        module list 2>&1 | grep -q ompi-cray \
          && module list 2>&1 | grep -q PrgEnv-cray \
          && module list 2>&1 | grep -q hdf5 \
          && module list 2>&1 | grep -q rocm
    	if [ $? -ne 0 ]; then 
          echo "modulefile required: ompi-cray hdf5"
          echo "do: 'module use /home/users/twhite/share/modulefiles"
          echo "    'module load ompi-cray hdf5'"
          exit 1
        fi 
  esac

fi


if [ "$1" == "run" ]; then
  
  case $2 in
    summit)
      if [ -z $LSB_JOBID ]; then
        echo "Job not started. Start an interactive job with, e.g.:"
        echo "  bsub -q debug -nnodes 1 -P <PROJ_ID> -W 1:00 -Is /bin/bash"
        exit 1
      fi
      $0 build $2
      ;;
    poplar)
      #-- Rely on `srun` to submit job immediately
      #if [ -z $SLURM_JOBID ]; then
      #  echo "Job not started. Start an interactive job with, e.g.:"
      #  echo "  salloc --nodes=1 -p amdMI60 --time=02:00:00"
      #  exit 1
      #fi
      $0 build $2
  esac
  
fi

exit 0
