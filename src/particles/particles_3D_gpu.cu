#if defined(PARTICLES) && defined(PARTICLES_GPU)

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"../io.h"
#include"../global.h"
#include"../global_cuda.h"
#include "particles_3D.h"



void Particles_3D::Free_GPU_Array_Real( Real *array ){ cudaFree(array); }
void Particles_3D::Free_GPU_Array_int( int *array )  { cudaFree(array); }
void Particles_3D::Free_GPU_Array_bool( bool *array ){ cudaFree(array); }

void Particles_3D::Allocate_Particles_GPU_Array_Real( Real **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  #endif
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_Grid_Field_Real( Real **array_dev, int size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  #endif
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_GPU_Array_int( int **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  #endif
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(int)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_GPU_Array_bool( bool **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  #ifdef PRINT_GPU_MEMORY
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  #endif
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
  Set_Particles_Array_Real_Kernel<<<dim1dGrid,dim1dBlock>>>( value, array_dev, size);
  CudaCheckError();
}








#endif//PARTICLES