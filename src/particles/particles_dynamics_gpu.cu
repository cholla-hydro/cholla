#if defined(PARTICLES) && defined(PARTICLES_GPU)

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>
  #include <unistd.h>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/gpu.hpp"
  #include "particles_3D.h"

  #ifdef COSMOLOGY
    #include "../cosmology/cosmology.h"
// #include "../cosmology/cosmology_functions_gpu.h"

// FUTURE FIX: The Hubble function was defined here because I couldn't get it
// form other file, tried -dc flag when compiling buu paris broke.
__device__ Real Get_Hubble_Parameter_dev(Real a, Real H0, Real Omega_M, Real Omega_L, Real Omega_K)
{
  Real a2     = a * a;
  Real a3     = a2 * a;
  Real factor = (Omega_M / a3 + Omega_K / a2 + Omega_L);
  return H0 * sqrt(factor);
}
  #endif

__global__ void Calc_Particles_dti_Kernel(part_int_t n_local, Real dx, Real dy, Real dz, Real *vel_x_dev,
                                          Real *vel_y_dev, Real *vel_z_dev, Real *dti_array)
{
  __shared__ Real max_dti[TPB_PARTICLES];

  part_int_t id;
  int tid;

  // get a global thread ID
  id = blockIdx.x * blockDim.x + threadIdx.x;
  // and a thread id within the block
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();

  Real vx, vy, vz;

  // if( tid == 0 ) printf("%f  %f  %f \n", dx, dy, dz );

  // threads corresponding to real cells do the calculation
  if (id < n_local) {
    // every thread collects the variables it needs from global memory
    vx           = vel_x_dev[id];
    vy           = vel_y_dev[id];
    vz           = vel_z_dev[id];
    max_dti[tid] = fmax(fabs(vx) / dx, fabs(vy) / dy);
    max_dti[tid] = fmax(max_dti[tid], fabs(vz) / dz);
    max_dti[tid] = fmax(max_dti[tid], 0.0);
  }
  __syncthreads();

  // do the reduction in shared memory (find the max inverse timestep in the
  // block)
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      max_dti[tid] = fmax(max_dti[tid], max_dti[tid + s]);
    }
    __syncthreads();
  }

  // write the result for this block to global memory
  if (tid == 0) {
    dti_array[blockIdx.x] = max_dti[0];
  }
}

Real Particles_3D::Calc_Particles_dt_GPU_function(int ngrid, part_int_t n_particles_local, Real dx, Real dy, Real dz,
                                                  Real *vel_x, Real *vel_y, Real *vel_z, Real *dti_array_host,
                                                  Real *dti_array_dev)
{
  // // set values for GPU kernels
  // int ngrid =  (Particles.n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // printf("%f %f %f \n", dx, dy, dz);

  // Only runs if there are local particles
  if (ngrid == 0) {
    return 0;
  }

  hipLaunchKernelGGL(Calc_Particles_dti_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_particles_local, dx, dy, dz, vel_x,
                     vel_y, vel_z, dti_array_dev);
  CudaCheckError();

  // Initialize dt values
  Real max_dti = 0;
  // copy the dti array onto the CPU
  CudaSafeCall(cudaMemcpy(dti_array_host, dti_array_dev, ngrid * sizeof(Real), cudaMemcpyDeviceToHost));
  // find maximum inverse timestep from CFL condition
  for (int i = 0; i < ngrid; i++) {
    max_dti = fmax(max_dti, dti_array_host[i]);
  }

  return max_dti;
}

__global__ void Advance_Particles_KDK_Step1_Kernel(part_int_t n_local, Real dt, Real *pos_x_dev, Real *pos_y_dev,
                                                   Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev, Real *vel_z_dev,
                                                   Real *grav_x_dev, Real *grav_y_dev, Real *grav_z_dev)
{
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_local) {
    return;
  }

  // Advance velocities by half a step
  vel_x_dev[tid] += 0.5 * dt * grav_x_dev[tid];
  vel_y_dev[tid] += 0.5 * dt * grav_y_dev[tid];
  vel_z_dev[tid] += 0.5 * dt * grav_z_dev[tid];

  // Advance Positions using advanced velocities
  pos_x_dev[tid] += dt * vel_x_dev[tid];
  pos_y_dev[tid] += dt * vel_y_dev[tid];
  pos_z_dev[tid] += dt * vel_z_dev[tid];
}

__global__ void Advance_Particles_KDK_Step2_Kernel(part_int_t n_local, Real dt, Real *vel_x_dev, Real *vel_y_dev,
                                                   Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev,
                                                   Real *grav_z_dev)
{
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_local) {
    return;
  }

  // Advance velocities by the second half a step
  vel_x_dev[tid] += 0.5 * dt * grav_x_dev[tid];
  vel_y_dev[tid] += 0.5 * dt * grav_y_dev[tid];
  vel_z_dev[tid] += 0.5 * dt * grav_z_dev[tid];
}

void Particles_3D::Advance_Particles_KDK_Step1_GPU_function(part_int_t n_local, Real dt, Real *pos_x_dev,
                                                            Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev,
                                                            Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev,
                                                            Real *grav_y_dev, Real *grav_z_dev)
{
  // set values for GPU kernels
  int ngrid = (n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Only runs if there are local particles
  if (n_local > 0) {
    hipLaunchKernelGGL(Advance_Particles_KDK_Step1_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_local, dt, pos_x_dev,
                       pos_y_dev, pos_z_dev, vel_x_dev, vel_y_dev, vel_z_dev, grav_x_dev, grav_y_dev, grav_z_dev);
    CudaCheckError();
  }
}

void Particles_3D::Advance_Particles_KDK_Step2_GPU_function(part_int_t n_local, Real dt, Real *vel_x_dev,
                                                            Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev,
                                                            Real *grav_y_dev, Real *grav_z_dev)
{
  // set values for GPU kernels
  int ngrid = (n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Only runs if there are local particles
  if (n_local > 0) {
    hipLaunchKernelGGL(Advance_Particles_KDK_Step2_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_local, dt, vel_x_dev,
                       vel_y_dev, vel_z_dev, grav_x_dev, grav_y_dev, grav_z_dev);
    CudaCheckError();
  }
}

  #ifdef COSMOLOGY

__global__ void Advance_Particles_KDK_Step1_Cosmo_Kernel(part_int_t n_local, Real da, Real *pos_x_dev, Real *pos_y_dev,
                                                         Real *pos_z_dev, Real *vel_x_dev, Real *vel_y_dev,
                                                         Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev,
                                                         Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h,
                                                         Real Omega_M, Real Omega_L, Real Omega_K)
{
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_local) {
    return;
  }

  Real vel_x, vel_y, vel_z;
  vel_x = vel_x_dev[tid];
  vel_y = vel_y_dev[tid];
  vel_z = vel_z_dev[tid];

  Real da_half, a_half, H, H_half, dt, dt_half;
  da_half = da / 2;
  a_half  = current_a + da_half;

  H      = Get_Hubble_Parameter_dev(current_a, H0, Omega_M, Omega_L, Omega_K);
  H_half = Get_Hubble_Parameter_dev(a_half, H0, Omega_M, Omega_L, Omega_K);

  dt      = da / (current_a * H) * cosmo_h;
  dt_half = da / (a_half * H_half) * cosmo_h / (a_half);

  // if ( tid == 0 ) printf( "dt: %f\n", dt);
  // if ( tid == 0 ) printf( "pos_x: %f\n", pos_x_dev[tid]);
  // if ( tid == 0 ) printf( "vel_x: %f\n", vel_x_dev[tid]);
  // if ( tid == 0 ) printf( "grav_x: %f\n", grav_x_dev[tid]);

  // Advance velocities by half a step
  vel_x          = (current_a * vel_x + 0.5 * dt * grav_x_dev[tid]) / a_half;
  vel_y          = (current_a * vel_y + 0.5 * dt * grav_y_dev[tid]) / a_half;
  vel_z          = (current_a * vel_z + 0.5 * dt * grav_z_dev[tid]) / a_half;
  vel_x_dev[tid] = vel_x;
  vel_y_dev[tid] = vel_y;
  vel_z_dev[tid] = vel_z;

  // Advance Positions using advanced velocities
  pos_x_dev[tid] += dt_half * vel_x;
  pos_y_dev[tid] += dt_half * vel_y;
  pos_z_dev[tid] += dt_half * vel_z;
}

__global__ void Advance_Particles_KDK_Step2_Cosmo_Kernel(part_int_t n_local, Real da, Real *vel_x_dev, Real *vel_y_dev,
                                                         Real *vel_z_dev, Real *grav_x_dev, Real *grav_y_dev,
                                                         Real *grav_z_dev, Real current_a, Real H0, Real cosmo_h,
                                                         Real Omega_M, Real Omega_L, Real Omega_K)
{
  part_int_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_local) {
    return;
  }

  Real vel_x, vel_y, vel_z;
  vel_x = vel_x_dev[tid];
  vel_y = vel_y_dev[tid];
  vel_z = vel_z_dev[tid];

  Real da_half, a_half, dt;
  da_half = da / 2;
  a_half  = current_a - da_half;

  dt = da / (current_a * Get_Hubble_Parameter_dev(current_a, H0, Omega_M, Omega_L, Omega_K)) * cosmo_h;

  // Advance velocities by the second half a step
  vel_x_dev[tid] = (a_half * vel_x + 0.5 * dt * grav_x_dev[tid]) / current_a;
  vel_y_dev[tid] = (a_half * vel_y + 0.5 * dt * grav_y_dev[tid]) / current_a;
  vel_z_dev[tid] = (a_half * vel_z + 0.5 * dt * grav_z_dev[tid]) / current_a;
}

void Particles_3D::Advance_Particles_KDK_Step1_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *pos_x_dev,
                                                                  Real *pos_y_dev, Real *pos_z_dev, Real *vel_x_dev,
                                                                  Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev,
                                                                  Real *grav_y_dev, Real *grav_z_dev, Real current_a,
                                                                  Real H0, Real cosmo_h, Real Omega_M, Real Omega_L,
                                                                  Real Omega_K)
{
  // set values for GPU kernels
  int ngrid = (n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Only runs if there are local particles
  if (n_local > 0) {
    hipLaunchKernelGGL(Advance_Particles_KDK_Step1_Cosmo_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_local, delta_a,
                       pos_x_dev, pos_y_dev, pos_z_dev, vel_x_dev, vel_y_dev, vel_z_dev, grav_x_dev, grav_y_dev,
                       grav_z_dev, current_a, H0, cosmo_h, Omega_M, Omega_L, Omega_K);
    CHECK(cudaDeviceSynchronize());
    // CudaCheckError();
  }
}

void Particles_3D::Advance_Particles_KDK_Step2_Cosmo_GPU_function(part_int_t n_local, Real delta_a, Real *vel_x_dev,
                                                                  Real *vel_y_dev, Real *vel_z_dev, Real *grav_x_dev,
                                                                  Real *grav_y_dev, Real *grav_z_dev, Real current_a,
                                                                  Real H0, Real cosmo_h, Real Omega_M, Real Omega_L,
                                                                  Real Omega_K)
{
  // set values for GPU kernels
  int ngrid = (n_local - 1) / TPB_PARTICLES + 1;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB_PARTICLES, 1, 1);

  // Only runs if there are local particles
  if (n_local > 0) {
    hipLaunchKernelGGL(Advance_Particles_KDK_Step2_Cosmo_Kernel, dim1dGrid, dim1dBlock, 0, 0, n_local, delta_a,
                       vel_x_dev, vel_y_dev, vel_z_dev, grav_x_dev, grav_y_dev, grav_z_dev, current_a, H0, cosmo_h,
                       Omega_M, Omega_L, Omega_K);
    CHECK(cudaDeviceSynchronize());
    // CudaCheckError();
  }
}

  #endif  // COSMOLOGY

#endif
