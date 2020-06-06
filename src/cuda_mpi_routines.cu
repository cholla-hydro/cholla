#include<stdio.h>
#include"gpu.hpp"
#include"io.h"
#include"cuda_mpi_routines.h"

/*! \fn int initialize_cuda_mpi(int myid, int nprocs);
 *  \brief CUDA initialization within MPI. */
int initialize_cuda_mpi(int myid, int nprocs)
{
  int i_device = 0;   //GPU device for this process
  int n_device;   //number of GPU devices available

  cudaError_t flag_error;

  //get the number of cuda devices    
  flag_error = cudaGetDeviceCount(&n_device);

  //check for errors
  if(flag_error!=cudaSuccess)
  {
    if(flag_error==cudaErrorNoDevice)
      fprintf(stderr,"cudaGetDeviceCount: Error! for myid = %d and n_device = %d; cudaErrorNoDevice\n",myid,n_device);
    if(flag_error==cudaErrorInsufficientDriver)
      fprintf(stderr,"cudaGetDeviceCount: Error! for myid = %d and n_device = %d; cudaErrorInsufficientDriver\n",myid,n_device);
    fflush(stderr);
    return 1;
  }

  //set a cuda device for each process
  cudaSetDevice(myid%n_device);

  //double check
  cudaGetDevice(&i_device);

  // printf("In initialize_cuda_mpi: myid = %d, i_device = %d, n_device = %d\n",myid,i_device,n_device);
  // fflush(stdout);

  return 0;
    
}
