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


void Particles_3D::Allocate_Particles_Field_Real( Real **array_dev, part_int_t size ){
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Copy_Particle_Field_Real_Host_to_Device( Real *array_host, Real *array_dev, part_int_t size){
  CudaSafeCall( cudaMemcpy(array_dev, array_host, size*sizeof(Real), cudaMemcpyHostToDevice) );
  cudaDeviceSynchronize();
}

void Particles_3D::Copy_Particle_Field_Real_Device_to_Host( Real *array_dev, Real *array_host, part_int_t size){
  CudaSafeCall( cudaMemcpy(array_host, array_dev, size*sizeof(Real), cudaMemcpyDeviceToHost) );
  cudaDeviceSynchronize();
}



__global__ void Set_Particle_Field_Real_Kernel( Real value, Real *array_dev, part_int_t size ){
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid < size ) array_dev[tid] = value;
}



void Particles_3D::Set_Particle_Field_Real( Real value, Real *array_dev, part_int_t size){
  
  // set values for GPU kernels
  int ngrid =  (size + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  Set_Particle_Field_Real_Kernel<<<dim1dGrid,dim1dBlock>>>( value, array_dev, size);
  CudaCheckError();
}



void Particles_3D::Allocate_Memory_GPU(){
  CudaSafeCall( cudaMalloc((void**)&G.density_dev,  G.n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&G.gravity_x_dev,  G.n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&G.gravity_y_dev,  G.n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&G.gravity_z_dev,  G.n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&G.potential_dev,  G.n_cells_potential*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&G.dti_array_dev,  G.size_dt_array*sizeof(Real)) );
    
  chprintf( " Allocated GPU memory.\n");  
}


void Particles_3D::Free_Memory_GPU(){
  
  cudaFree(G.density_dev);
  cudaFree(G.potential_dev);
  cudaFree(G.gravity_x_dev);
  cudaFree(G.gravity_y_dev);
  cudaFree(G.gravity_z_dev);
  cudaFree(G.dti_array_dev);
  
  cudaFree(pos_x_dev);
  cudaFree(pos_y_dev);
  cudaFree(pos_z_dev);
  cudaFree(vel_x_dev);
  cudaFree(vel_y_dev);
  cudaFree(vel_z_dev);
  cudaFree(grav_x_dev);
  cudaFree(grav_y_dev);
  cudaFree(grav_z_dev);
  
}




#endif//PARTICLES