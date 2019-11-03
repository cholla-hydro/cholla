#if defined(PARTICLES) && defined(PARTICLES_GPU) 

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"../global.h"
#include"../global_cuda.h"
#include "particles_3D.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ void Get_Indexes_CIC( Real xMin, Real yMin, Real zMin, Real dx, Real dy, Real dz, Real pos_x, Real pos_y, Real pos_z, int &indx_x, int &indx_y, int &indx_z ){
  indx_x = (int) floor( ( pos_x - xMin - 0.5*dx ) / dx );
  indx_y = (int) floor( ( pos_y - yMin - 0.5*dy ) / dy );
  indx_z = (int) floor( ( pos_z - zMin - 0.5*dz ) / dz );
}

__global__ void Get_Density_CIC_Kernel( part_int_t n_local, Real particle_mass,  Real *density_dev, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real xMin, Real yMin, Real zMin, Real xMax, Real yMax, Real zMax, Real dx, Real dy, Real dz, int nx, int ny, int nz, int n_ghost  ){
  
  int tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  int nx_g, ny_g;
  nx_g = nx + 2*n_ghost;
  ny_g = ny + 2*n_ghost;
  
  Real pos_x, pos_y, pos_z, pMass;
  Real cell_center_x, cell_center_y, cell_center_z;
  Real delta_x, delta_y, delta_z;
  Real dV_inv = 1./(dx*dy*dz);
  
  pos_x = pos_x_dev[tid];
  pos_y = pos_y_dev[tid];
  pos_z = pos_z_dev[tid]; 
  // 
  
  #ifdef SINGLE_PARTICLE_MASS
  pMass = particle_mass * dV_inv;
  #else
  pMass = mass[pIndx] * dV_inv;
  #endif
  
  int indx_x, indx_y, indx_z, indx;
  Get_Indexes_CIC( xMin, yMin, zMin, dx, dy, dz, pos_x, pos_y, pos_z, indx_x, indx_y, indx_z );
  
  bool in_local = true;
  
  if ( pos_x < xMin || pos_x >= xMax ) in_local = false;
  if ( pos_y < yMin || pos_y >= yMax ) in_local = false;
  if ( pos_z < zMin || pos_z >= zMax ) in_local = false;
  if ( ! in_local  ) {
    printf(" Density CIC Error: Particle outside local domain [%f  %f  %f]  [%f %f] [%f %f] [%f %f]\n ", pos_x, pos_y, pos_z, xMin, xMax, yMin, yMax, zMin, zMax);
    return;
  }
  
  cell_center_x = xMin + indx_x*dx + 0.5*dx;
  cell_center_y = yMin + indx_y*dy + 0.5*dy;
  cell_center_z = zMin + indx_z*dz + 0.5*dz;
  delta_x = 1 - ( pos_x - cell_center_x ) / dx;
  delta_y = 1 - ( pos_y - cell_center_y ) / dy;
  delta_z = 1 - ( pos_z - cell_center_z ) / dz;
  indx_x += n_ghost;
  indx_y += n_ghost;
  indx_z += n_ghost;
  
  
  indx = indx_x + indx_y*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * delta_y * delta_z;
  atomicAdd( &density_dev[indx],  pMass  * delta_x * delta_y * delta_z);
  
  indx = (indx_x+1) + indx_y*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * delta_y * delta_z;
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * delta_y * delta_z);
  
  indx = indx_x + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * (1-delta_y) * delta_z;
  atomicAdd( &density_dev[indx], pMass  * delta_x * (1-delta_y) * delta_z);
  // 
  indx = indx_x + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * delta_y * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * delta_x * delta_y * (1-delta_z) );
  
  indx = (indx_x+1) + (indx_y+1)*nx_g + indx_z*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * (1-delta_y) * delta_z;
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * (1-delta_y) * delta_z);
  
  indx = (indx_x+1) + indx_y*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * (1-delta_x) * delta_y * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * (1-delta_x) * delta_y * (1-delta_z));
  
  indx = indx_x + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass  * delta_x * (1-delta_y) * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass  * delta_x * (1-delta_y) * (1-delta_z));
  
  indx = (indx_x+1) + (indx_y+1)*nx_g + (indx_z+1)*nx_g*ny_g;
  // density_dev[indx] += pMass * (1-delta_x) * (1-delta_y) * (1-delta_z);
  atomicAdd( &density_dev[indx], pMass * (1-delta_x) * (1-delta_y) * (1-delta_z));

}




void Particles_3D::Clear_Density_GPU_function( Real *density_dev, int n_cells){
  Set_Particle_Field_Real( 0.0, density_dev, n_cells);  
}

void Get_Density_CIC_GPU_function(part_int_t n_local, Real particle_mass,  Real xMin, Real xMax, Real yMin, Real yMax, Real zMin, Real zMax, Real dx, Real dy, Real dz, int nx_local, int ny_local, int nz_local, int n_ghost_particles_grid, int n_cells, Real *density_h, Real *density_dev, Real *pos_x_dev, Real *pos_y_dev , Real *pos_z_dev){
    
  // set values for GPU kernels
  int ngrid =  (n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  Get_Density_CIC_Kernel<<<dim1dGrid,dim1dBlock>>>( n_local, particle_mass, density_dev, pos_x_dev, pos_y_dev, pos_z_dev, xMin, yMin, zMin, xMax, yMax, zMax, dx, dy, dz, nx_local, ny_local, nz_local, n_ghost_particles_grid );
  CudaCheckError();
  cudaDeviceSynchronize();
  
  //Copy the density from device to host
  CudaSafeCall( cudaMemcpy(density_h, density_dev, n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );  
  
}

#endif