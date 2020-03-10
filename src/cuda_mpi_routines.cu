#include<stdio.h>
#include<hip/hip_runtime.h>
#include<hip/hip_runtime.h>
#include"io.h"
#include"cuda_mpi_routines.h"

/*! \fn int initialize_cuda_mpi(int myid, int nprocs);
 *  \brief CUDA initialization within MPI. */
int initialize_cuda_mpi(int myid, int nprocs)
{
  int i_device = 0;   //GPU device for this process
  int n_device;   //number of GPU devices available

  hipError_t flag_error;

  //get the number of cuda devices    
  flag_error = hipGetDeviceCount(&n_device);

  //check for errors
  if(flag_error!=hipSuccess)
  {
    if(flag_error==hipErrorNoDevice)
      fprintf(stderr,"hipGetDeviceCount: Error! for myid = %d and n_device = %d; hipErrorNoDevice\n",myid,n_device);
    if(flag_error==hipErrorInsufficientDriver)
      fprintf(stderr,"hipGetDeviceCount: Error! for myid = %d and n_device = %d; hipErrorInsufficientDriver\n",myid,n_device);
    fflush(stderr);
    return 1;
  }

  //set a cuda device for each process
  hipSetDevice(myid%n_device);

  //double check
  hipGetDevice(&i_device);

  //printf("In initialize_cuda_mpi: myid = %d, i_device = %d, n_device = %d\n",myid,i_device,n_device);
  //fflush(stdout);

  return 0;
    
}
