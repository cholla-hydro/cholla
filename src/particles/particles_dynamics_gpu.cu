#if defined(PARTICLES) && defined(PARTICLES_GPU) 

#include <unistd.h>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"../global.h"
#include"../global_cuda.h"
#include"../grid3D.h"
#include"../io.h"
#include "particles_3D.h"

#ifdef COSMOLOGY
#include"../cosmology/cosmology.h"
#include"../cosmology/cosmology_functions_gpu.h"
#endif


__global__ void Calc_Particles_dti_Kernel( part_int_t n_local, Real dx, Real dy, Real dz, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *dti_array )
{
  __shared__ Real max_dti[TPB_PARTICLES];
  
  part_int_t id;
  int tid;
   
  // get a global thread ID
  id = blockIdx.x * blockDim.x + threadIdx.x ;
  // and a thread id within the block  
  tid = threadIdx.x;
  
  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();
  
  Real vx, vy, vz;
  
  // threads corresponding to real cells do the calculation
  if (id < n_local ){
    // every thread collects the variables it needs from global memory
    vx =  vel_x_dev[id];
    vy =  vel_y_dev[id];
    vz =  vel_z_dev[id];
    max_dti[tid] = fmax( fabs(vx)/dx, fabs(vy)/dy);
    max_dti[tid] = fmax( max_dti[tid], fabs(vz)/dz);
    max_dti[tid] = fmax( max_dti[tid], 0.0);
  }
  __syncthreads();
  
  // do the reduction in shared memory (find the max inverse timestep in the block)
  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      max_dti[tid] = fmax(max_dti[tid], max_dti[tid + s]);
    }
    __syncthreads();
  }
  
  // write the result for this block to global memory
  if (tid == 0) dti_array[blockIdx.x] = max_dti[0];

}





Real Grid3D::Calc_Particles_dt_GPU(){
  
  
  // set values for GPU kernels
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  if ( ngrid > Particles.G.size_blocks_array ) chprintf(" Error: particles dt_array too small\n");

  Calc_Particles_dti_Kernel<<<dim1dGrid,dim1dBlock>>>( Particles.n_local, Particles.G.dx, Particles.G.dy, Particles.G.dz, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.G.dti_array_dev );
  CudaCheckError();
  
  // Initialize dt values 
  Real max_dti = 0; 
  // copy the dti array onto the CPU
  CudaSafeCall( cudaMemcpy(Particles.G.dti_array_host, Particles.G.dti_array_dev, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // find maximum inverse timestep from CFL condition
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, Particles.G.dti_array_host[i]);
  }
  
  Real dt_min;
  
  #ifdef COSMOLOGY
  Real scale_factor, vel_factor, da_min;
  scale_factor = 1 / ( Cosmo.current_a * Cosmo.Get_Hubble_Parameter( Cosmo.current_a) ) * Cosmo.cosmo_h;
  vel_factor = Cosmo.current_a / scale_factor;
  da_min = vel_factor / max_dti;
  dt_min = Cosmo.Get_dt_from_da( da_min );
  #else
  dt_min = 1 / max_dti;
  #endif
  
  return Particles.C_cfl*dt_min;

}



__global__ void Advance_Particles_KDK_Step1_Kernel( part_int_t n_local, Real dt, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev ){
  
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  // Advance velocities by half a step
  vel_x_dev[tid] += 0.5 * dt * grav_x_dev[tid];
  vel_y_dev[tid] += 0.5 * dt * grav_y_dev[tid];
  vel_z_dev[tid] += 0.5 * dt * grav_z_dev[tid];
  
  //Advance Posiotions using advanced velocities
  pos_x_dev[tid] += dt * vel_x_dev[tid];
  pos_y_dev[tid] += dt * vel_y_dev[tid];
  pos_z_dev[tid] += dt * vel_z_dev[tid];
}


__global__ void Advance_Particles_KDK_Step2_Kernel( part_int_t n_local, Real dt, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev ){
  
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  // Advance velocities by the second half a step
  vel_x_dev[tid] += 0.5 * dt * grav_x_dev[tid];
  vel_y_dev[tid] += 0.5 * dt * grav_y_dev[tid];
  vel_z_dev[tid] += 0.5 * dt * grav_z_dev[tid];
  
}

__global__ void Advance_Particles_KDK_Step1_Cosmo_Kernel( part_int_t n_local, Real da, Real *pos_x_dev, Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h, Real Omega_M, Real Omega_L, Real Omega_K ){
  
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  Real vel_x, vel_y, vel_z;
  vel_x = vel_x_dev[tid];
  vel_y = vel_y_dev[tid];
  vel_z = vel_z_dev[tid];
  
  
  Real da_half, a_half, H, H_half, dt, dt_half;
  da_half = da/2;
  a_half = current_a + da_half;
  
  H = Get_Hubble_Parameter_dev( current_a, H0, Omega_M, Omega_L, Omega_K );
  H_half = Get_Hubble_Parameter_dev( a_half, H0, Omega_M, Omega_L, Omega_K );
  
  dt = da / ( current_a * H ) * cosmo_h;
  dt_half = da / ( a_half * H_half ) * cosmo_h / ( a_half );
  
  // if ( tid == 0 ) printf( "dt: %f\n", dt);
  // if ( tid == 0 ) printf( "pos_x: %f\n", pos_x_dev[tid]);
  // if ( tid == 0 ) printf( "vel_x: %f\n", vel_x_dev[tid]);
  // if ( tid == 0 ) printf( "grav_x: %f\n", grav_x_dev[tid]);
  
  // Advance velocities by half a step
  vel_x = ( current_a*vel_x + 0.5*dt*grav_x_dev[tid] ) / a_half;
  vel_y = ( current_a*vel_y + 0.5*dt*grav_y_dev[tid] ) / a_half;
  vel_z = ( current_a*vel_z + 0.5*dt*grav_z_dev[tid] ) / a_half;
  vel_x_dev[tid] = vel_x;
  vel_y_dev[tid] = vel_y;
  vel_z_dev[tid] = vel_z;  
  
  //Advance Posiotions using advanced velocities
  pos_x_dev[tid] += dt_half * vel_x;
  pos_y_dev[tid] += dt_half * vel_y;
  pos_z_dev[tid] += dt_half * vel_z;
}


__global__ void Advance_Particles_KDK_Step2_Cosmo_Kernel( part_int_t n_local, Real da, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h, Real Omega_M, Real Omega_L, Real Omega_K ){
  
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x ;
  if ( tid >= n_local) return;
  
  Real vel_x, vel_y, vel_z;
  vel_x = vel_x_dev[tid];
  vel_y = vel_y_dev[tid];
  vel_z = vel_z_dev[tid];
  
  Real da_half, a_half, dt;
  da_half = da/2;
  a_half = current_a - da_half;
  
  dt = da / ( current_a * Get_Hubble_Parameter_dev( current_a, H0, Omega_M, Omega_L, Omega_K ) ) * cosmo_h;
  
  // Advance velocities by the second half a step
  vel_x_dev[tid] = ( a_half*vel_x + 0.5*dt*grav_x_dev[tid] ) / current_a;
  vel_y_dev[tid] = ( a_half*vel_y + 0.5*dt*grav_y_dev[tid] ) / current_a;
  vel_z_dev[tid] = ( a_half*vel_z + 0.5*dt*grav_z_dev[tid] ) / current_a;
  
}

void Grid3D::Advance_Particles_KDK_Step1_GPU(){
  
  // set values for GPU kernels
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  #ifdef COSMOLOGY
  Advance_Particles_KDK_Step1_Cosmo_Kernel<<<dim1dGrid,dim1dBlock>>>( Particles.n_local, Cosmo.delta_a, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev, Cosmo.current_a, Cosmo.H0, Cosmo.cosmo_h, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K );
  CudaCheckError();
  #else
  Advance_Particles_KDK_Step1_Kernel<<<dim1dGrid,dim1dBlock>>>( Particles.n_local, Particles.dt, Particles.pos_x_dev, Particles.pos_y_dev, Particles.pos_z_dev, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev );
  CudaCheckError();
  #endif//COSMOLOGY
}

void Grid3D::Advance_Particles_KDK_Step2_GPU(){
  
  // set values for GPU kernels
  int ngrid =  (Particles.n_local + TPB_PARTICLES - 1) / TPB_PARTICLES;
  // number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block   
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);
  
  
  #ifdef COSMOLOGY
  Advance_Particles_KDK_Step2_Cosmo_Kernel<<<dim1dGrid,dim1dBlock>>>( Particles.n_local, Cosmo.delta_a, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev, Cosmo.current_a, Cosmo.H0, Cosmo.cosmo_h, Cosmo.Omega_M, Cosmo.Omega_L, Cosmo.Omega_K );
  CudaCheckError();
  #else
  Advance_Particles_KDK_Step2_Kernel<<<dim1dGrid,dim1dBlock>>>( Particles.n_local, Particles.dt, Particles.vel_x_dev, Particles.vel_y_dev, Particles.vel_z_dev, Particles.grav_x_dev, Particles.grav_y_dev, Particles.grav_z_dev );
  CudaCheckError();
  #endif//COSMOLOGY
  
  
}


#endif