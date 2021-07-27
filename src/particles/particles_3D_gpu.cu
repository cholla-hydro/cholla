#if defined(PARTICLES) && defined(PARTICLES_GPU)

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"gpu.hpp"
#include"../io.h"
#include"../global.h"
#include"../global_cuda.h"
#include "particles_3D.h"



void Particles_3D::Free_GPU_Array_Real( Real *array ){ cudaFree(array); }
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

void Particles_3D::Reallocate_and_Copy_Partciles_Array_Real( Real **src_array_dev, part_int_t size_initial, part_int_t size_end  ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  cudaDeviceSynchronize();
  #ifdef PRINT_GPU_MEMORY
  printf( "ReAlocating GPU Memory:  %ld  MB free \n", global_free/1000000);
  #endif
  if ( global_free < size_end*sizeof(Real) ){
    printf( "ERROR: Not enough global device memory \n" );
    printf( " Available Memory: %ld  MB \n", global_free/1000000  );
    printf( " Requested Memory: %ld  MB \n", size_end*sizeof(Real)/1000000  );
    exit(-1);
  }
  Real *temp_array_dev;
  CudaSafeCall( cudaMalloc((void**)&temp_array_dev,  size_end*sizeof(Real)) );
  cudaDeviceSynchronize();
  // printf( " Alocated GPU Memory:  %ld  MB \n", size_end*sizeof(Real)/1000000 );
  if ( size_initial*sizeof(Real) > size_end*sizeof(Real) ){
    printf("ERROR: Memory to copy larger than array size\n" );
    exit(-1);
  }
  // printf( " Copying:  %ld  ->  %ld  \n", size_initial*sizeof(Real), size_end*sizeof(Real) );
  // CudaSafeCall( cudaMemcpy(temp_array_dev, *src_array_dev, size_initial*sizeof(Real), cudaMemcpyDeviceToDevice) );
  // NOTE: cudaMemcpy is not working! made kernel to do the device to device copy
  Copy_Device_to_Device( *src_array_dev, temp_array_dev,  size_initial );
  cudaDeviceSynchronize();
  CudaSafeCall( cudaFree( *src_array_dev ));
  cudaDeviceSynchronize();
  *src_array_dev = temp_array_dev;
  
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








#endif//PARTICLES