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
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_Grid_Field_Real( Real **array_dev, int size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free/1000000);
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(Real)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_Field_int( int **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free*1e-6/1000000);
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(int)) );
  cudaDeviceSynchronize();
}

void Particles_3D::Allocate_Particles_Field_bool( bool **array_dev, part_int_t size ){
  size_t global_free, global_total;
  CudaSafeCall( cudaMemGetInfo( &global_free, &global_total ) );
  chprintf( "Alocating GPU Memory:  %d  MB free \n", global_free*1e-6/1000000);
  CudaSafeCall( cudaMalloc((void**)array_dev,  size*sizeof(bool)) );
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
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
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




#ifdef MPI_CHOLLA
void Particles_3D::Allocate_Memory_GPU_MPI(){
  Allocate_Particles_Field_bool( &transfer_particles_flags_x0, particles_buffer_size );
  Allocate_Particles_Field_bool( &transfer_particles_flags_x1, particles_buffer_size );
  Allocate_Particles_Field_bool( &transfer_particles_flags_y0, particles_buffer_size );
  Allocate_Particles_Field_bool( &transfer_particles_flags_y1, particles_buffer_size );
  Allocate_Particles_Field_bool( &transfer_particles_flags_z0, particles_buffer_size );
  Allocate_Particles_Field_bool( &transfer_particles_flags_z1, particles_buffer_size );

  Allocate_Particles_Field_int( &transfer_particles_indxs_x0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_indxs_x1, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_indxs_y0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_indxs_y1, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_indxs_z0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_indxs_z1, particles_buffer_size);

  Allocate_Particles_Field_int( &transfer_particles_partial_sum_x0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_partial_sum_x1, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_partial_sum_y0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_partial_sum_y1, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_partial_sum_z0, particles_buffer_size);
  Allocate_Particles_Field_int( &transfer_particles_partial_sum_z1, particles_buffer_size);
}
#endif //MPI_CHOLLA


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
  
  #ifdef MPI_CHOLLA
  cudaFree(transfer_particles_flags_x0);
  cudaFree(transfer_particles_flags_x1);
  cudaFree(transfer_particles_flags_y0);
  cudaFree(transfer_particles_flags_y1);
  cudaFree(transfer_particles_flags_z0);
  cudaFree(transfer_particles_flags_z1);
  
  cudaFree(transfer_particles_partial_sum_x0);
  cudaFree(transfer_particles_partial_sum_x1);
  cudaFree(transfer_particles_partial_sum_y0);
  cudaFree(transfer_particles_partial_sum_y1);
  cudaFree(transfer_particles_partial_sum_z0);
  cudaFree(transfer_particles_partial_sum_z1);
  
  cudaFree(transfer_particles_indxs_x0);
  cudaFree(transfer_particles_indxs_x1);
  cudaFree(transfer_particles_indxs_y0);
  cudaFree(transfer_particles_indxs_y1);
  cudaFree(transfer_particles_indxs_z0);
  cudaFree(transfer_particles_indxs_z1);
  #endif
}




#endif//PARTICLES