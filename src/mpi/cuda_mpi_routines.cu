#include <stdio.h>
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include "../mpi/cuda_mpi_routines.h"

// #define PRINT_DEVICE_IDS

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

  //get host name
  char pname[MPI_MAX_PROCESSOR_NAME];     //node hostname
  int  pname_length;          //length of node hostname
  MPI_Get_processor_name(pname, &pname_length);

  //set a cuda device for each process
  cudaSetDevice(myid%n_device);

  //double check
  cudaGetDevice(&i_device);

  #ifdef PRINT_DEVICE_IDS
  printf("In initialize_cuda_mpi: name:%s myid = %d, i_device = %d, n_device = %d\n",pname, myid,i_device,n_device);
  fflush(stdout);
  MPI_Barrier(world);
  #endif

  return 0;

}
