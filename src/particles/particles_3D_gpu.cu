#if defined(PARTICLES) 

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include "../io/io.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../particles/particles_3D.h"





void Particles_3D::Free_GPU_Array_Real( Real *array ){ cudaFree(array); }


void Particles_3D::Allocate_Particles_Grid_Field_Real( Real **array_dev, int size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Allocating GPU Memory:  %ld  MB free \n", global_free/1000000);
  #endif
  if ( global_free < size*sizeof(Real) ){
    printf( "ERROR: Not enough global device memory \n" );
    printf( " Available Memory: %ld  MB \n", global_free/1000000  );
    printf( " Requested Memory: %ld  MB \n", size*sizeof(Real)/1000000  );
    exit(-1);
  }
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}



#ifdef PARTICLES_GPU

#ifdef PRINT_MAX_MEMORY_USAGE
#include "../mpi/mpi_routines.h"

void Particles_3D::Print_Max_Memory_Usage(){
  
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  cudaDeviceSynchronize();
  
  part_int_t n_local_max, n_total, mem_usage;
  Real fraction_max, global_free_min;
  
  n_local_max = (part_int_t) ReduceRealMax( (Real) n_local );
  n_total = ReducePartIntSum( n_local );
  fraction_max = (Real) n_local_max / (Real) n_total;
  mem_usage = n_local_max * 9 * sizeof(Real); //Usage for pos, vel ans accel.
  
  global_free_min = ReduceRealMin( (Real) global_free  );
  
  chprintf( " Particles GPU Memory: N_local_max: %ld  (%.1f %)  mem_usage: %ld MB     global_free_min: %.1f MB  \n", n_local_max, fraction_max*100, mem_usage/1000000, global_free_min/1000000 );
  
  
}

#endif 



void Particles_3D::Free_GPU_Array_int( int *array )  { cudaFree(array); }
void Particles_3D::Free_GPU_Array_bool( bool *array ){ cudaFree(array); }


void __global__ Copy_Device_to_Device_Kernel( Real *src_array_dev, Real *dst_array_dev, part_int_t size ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid < size ) dst_array_dev[tid] = src_array_dev[tid];
}

void Copy_Device_to_Device( Real *src_array_dev, Real *dst_array_dev, part_int_t size ){
  int ngrid =  (size + TPB_PARTICLES - 1) / TPB_PARTICLES;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  hipLaunchKernelGGL(Copy_Device_to_Device_Kernel, dim1dGrid, dim1dBlock, 0, 0,  src_array_dev, dst_array_dev, size);
  CudaCheckError();

}


void Particles_3D::Allocate_Particles_GPU_Array_Real( Real **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Allocating GPU Memory:  %ld  MB free \n", global_free/1000000);
  #endif
  if ( global_free < size*sizeof(Real) ){
    printf( "ERROR: Not enough global device memory \n" );
    printf( " Available Memory: %ld  MB \n", global_free/1000000  );
    printf( " Requested Memory: %ld  MB \n", size*sizeof(Real)/1000000  );
    exit(-1);
  }
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_GPU_Array_int( int **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Allocating GPU Memory:  %ld  MB free \n", global_free/1000000);
  #endif
  if ( global_free < size*sizeof(int) ){
    printf( "ERROR: Not enough global device memory \n" );
    printf( " Available Memory: %ld  MB \n", global_free/1000000  );
    printf( " Requested Memory: %ld  MB \n", size*sizeof(int)/1000000  );
    exit(-1);
  }
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(int)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_GPU_Array_bool( bool **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Allocating GPU Memory:  %ld  MB free \n", global_free/1000000);
  #endif
  if ( global_free < size*sizeof(bool) ){
    printf( "ERROR: Not enough global device memory \n" );
    printf( " Available Memory: %ld  MB \n", global_free/1000000  );
    printf( " Requested Memory: %ld  MB \n", size*sizeof(bool)/1000000  );
    exit(-1);
  }
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(bool)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Copy_Particles_Array_Real_Host_to_Device( Real *array_host, Real *array_dev, part_int_t size){
  CudaSafeCall( cudaMemcpy(array_dev, array_host, size*sizeof(Real), cudaMemcpyHostToDevice) );
  cudaDeviceSynchronize();
}

void Particles_3D::Copy_Particles_Array_Real_Device_to_Host( Real *array_dev, Real *array_host, part_int_t size){
  CudaSafeCall( cudaMemcpy(array_host, array_dev, size*sizeof(Real), cudaMemcpyDeviceToHost) );
  cudaDeviceSynchronize();
}



__global__ void Set_Particles_Array_Real_Kernel( Real value, Real *array_dev, part_int_t size ){
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid < size ) array_dev[tid] = value;
}



void Particles_3D::Set_Particles_Array_Real( Real value, Real *array_dev, part_int_t size){

  // set values for GPU kernels
  int ngrid =  (size + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  hipLaunchKernelGGL(Set_Particles_Array_Real_Kernel, dim1dGrid, dim1dBlock, 0, 0,  value, array_dev, size);
  CudaCheckError();
}







#endif //PARTICLES_GPU
#endif//PARTICLES
