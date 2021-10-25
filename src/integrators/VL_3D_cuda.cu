/*! \file VL_3D_cuda.cu
 *  \brief Definitions of the cuda 3D VL algorithm functions. */

#ifdef CUDA
#ifdef VL

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/VL_3D_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../old_cholla/h_correction_3D_cuda.h"
#include "../old_cholla/subgrid_routines_3D.h"
#include "../io/io.h"
#include "../riemann_solvers/hll_cuda.h"

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields, Real density_floor);



void VL_Algorithm_3D_CUDA(Real *host_conserved0, Real *host_conserved1,
    Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
    int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound,
    Real ybound, Real zbound, Real dt, int n_fields, Real density_floor,
    Real U_floor, Real *host_grav_potential, Real max_dti_slow)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  //host_conserved0 contains the values at time n,
  //host_conserved1 will contain the values at time n+1

  if ( !block_size ) {
    // calculate the dimensions for the subgrid blocks
    sub_dimensions_3D(nx, ny, nz, n_ghost, &nx_s, &ny_s, &nz_s, &block1_tot, &block2_tot, &block3_tot, &remainder1, &remainder2, &remainder3, n_fields);
    //printf("Subgrid dimensions set: %d %d %d %d %d %d %d %d %d\n", nx_s, ny_s, nz_s, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3);
    //fflush(stdout);
    block_tot = block1_tot*block2_tot*block3_tot;
    // number of cells in one subgrid block
    BLOCK_VOL = nx_s*ny_s*nz_s;
    // dimensions for the 1D GPU grid
    ngrid = (BLOCK_VOL + TPB - 1) / TPB;
    #ifndef DYNAMIC_GPU_ALLOC
    block_size = true;
    #endif
  }
  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  // Set up pointers for the location to copy from and to
  if (block_tot == 1) {
    tmp1 = host_conserved0;
    tmp2 = host_conserved1;
    //host_grav_potential is NULL if not using GRAVITY
    temp_potential = host_grav_potential;
  }

  if ( !memory_allocated ){

    // allocate buffer to copy conserved variable blocks to/from
    if (block_tot > 1) {
      if ( cudaSuccess != cudaHostAlloc(&buffer, n_fields*BLOCK_VOL*sizeof(Real), cudaHostAllocDefault) ) {
        printf("Failed to allocate CPU buffer.\n");
      }
      tmp1 = buffer;
      tmp2 = buffer;

      #if defined( GRAVITY )
      if ( cudaSuccess != cudaHostAlloc(&buffer_potential, BLOCK_VOL*sizeof(Real), cudaHostAllocDefault) ) {
        printf("Failed to allocate CPU Grav_Potential buffer.\n");
      }
      #else
      buffer_potential = NULL;
      #endif
      temp_potential = buffer_potential;
    }
    // allocate an array on the CPU to hold max_dti returned from each thread block
    CudaSafeCall( cudaHostAlloc(&host_dti_array, ngrid*sizeof(Real), cudaHostAllocDefault) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaHostAlloc(&host_dt_array, ngrid*sizeof(Real), cudaHostAllocDefault) );
    #endif

    // allocate memory on the GPU
    //CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
    dev_conserved = d_conserved;
    CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_z,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif

    #if defined( GRAVITY )
    // CudaSafeCall( cudaMalloc((void**)&dev_grav_potential, BLOCK_VOL*sizeof(Real)) );
    dev_grav_potential = d_grav_potential;
    #else
    dev_grav_potential = NULL;
    #endif

    #ifndef DYNAMIC_GPU_ALLOC
    // If memory is single allocated: memory_allocated becomes true and successive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif
    //d_conserved = dev_conserved;
  }

  // counter for which block we're on
  int block = 0;


  // START LOOP OVER SUBGRID BLOCKS
  while (block < block_tot) {

    // copy the conserved variable block to the buffer
    host_copy_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved0, buffer, n_fields, host_grav_potential, buffer_potential);

    // calculate the global x, y, and z offsets of this subgrid block
    get_offsets_3D(nx_s, ny_s, nz_s, n_ghost, x_off, y_off, z_off, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, &x_off_s, &y_off_s, &z_off_s);

    // copy the conserved variables onto the GPU
    #ifndef HYDRO_GPU
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #endif

    #if defined( GRAVITY ) && !defined( GRAVITY_GPU )
    CudaSafeCall( cudaMemcpy(dev_grav_potential, temp_potential, BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #endif


    // Step 1: Use PCM reconstruction to put primitive variables into interface arrays
    hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();


    // Step 2: Calculate first-order upwind fluxes
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //ROE
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    #ifdef HLL
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    CudaCheckError();


    // Step 3: Update the conserved variables half a timestep
    hipLaunchKernelGGL(Update_Conserved_Variables_3D_half, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved_half, F_x, F_y, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, 0.5*dt, gama, n_fields, density_floor );
    CudaCheckError();


    // Step 4: Construct left and right interface values using updated conserved variables
    #ifdef PCM
    hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    #endif
    #ifdef PLMP
    hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PLMP
    #ifdef PLMC
    hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif
    #ifdef PPMP
    hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMP
    #ifdef PPMC
    hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMC
    CudaCheckError();


    // Step 5: Calculate the fluxes again
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //ROE
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    #ifdef HLL
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    CudaCheckError();

    #ifdef DE
    // Compute the divergence of Vel before updating the conserved array, this solves synchronization issues when adding this term on Update_Conserved_Variables_3D
    hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0,  dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz,  dt, gama, n_fields );
    CudaCheckError();
    #endif


    // Step 6: Update the conserved variable array
    hipLaunchKernelGGL(Update_Conserved_Variables_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, F_x, F_y, F_z, nx_s, ny_s, nz_s, x_off_s, y_off_s, z_off_s, n_ghost, dx, dy, dz, xbound, ybound, zbound, dt, gama, n_fields, density_floor, dev_grav_potential);
    CudaCheckError();

    #ifdef DE
    hipLaunchKernelGGL(Select_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields);
    hipLaunchKernelGGL(Sync_Energies_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #endif

    #ifdef TEMPERATURE_FLOOR
    hipLaunchKernelGGL(Apply_Temperature_Floor, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, U_floor );
    CudaCheckError();
    #endif //TEMPERATURE_FLOOR
    
    // add one to the counter
    block++;

  }


  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_VL_3D();
  #endif


  // return the maximum inverse timestep
  return;

}


void Free_Memory_VL_3D(){

  // free CPU memory
  if (block_tot > 1) CudaSafeCall( cudaFreeHost(buffer) );
  CudaSafeCall( cudaFreeHost(host_dti_array) );
  #ifdef COOLING_GPU
  CudaSafeCall( cudaFreeHost(host_dt_array) );
  #endif

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(Q_Lz);
  cudaFree(Q_Rz);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(F_z);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif
  #if defined( GRAVITY )
  cudaFree(dev_grav_potential);
  if (block_tot > 1) CudaSafeCall( cudaFreeHost(buffer_potential) );
  #endif

}

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields, Real density_floor )
{
  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  int n_cells = nx*ny*nz;

  // get a global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  int id = xid + yid*nx + zid*nx*ny;

  int imo = xid-1 + yid*nx + zid*nx*ny;
  int jmo = xid + (yid-1)*nx + zid*nx*ny;
  int kmo = xid + yid*nx + (zid-1)*nx*ny;

  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo, P, E, E_kin, GE;
  int ipo, jpo, kpo;
  #endif

  #ifdef DENSITY_FLOOR
  Real dens_0;
  #endif

  // threads corresponding to all cells except outer ring of ghost cells do the calculation
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1 && zid > 0 && zid < nz-1)
  {
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    //PRESSURE_DE
    E = dev_conserved[4*n_cells + id];
    GE = dev_conserved[(n_fields-1)*n_cells + id];
    E_kin = 0.5 * d * ( vx*vx + vy*vy + vz*vz );
    P = Get_Pressure_From_DE( E, E - E_kin, GE, gamma );
    P  = fmax(P, (Real) TINY_NUMBER);
    // P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before half step update.\n");
    //if (P < 0.0) printf("%d Negative pressure before half step update.\n", id);
    ipo = xid+1 + yid*nx + zid*nx*ny;
    jpo = xid + (yid+1)*nx + zid*nx*ny;
    kpo = xid + yid*nx + (zid+1)*nx*ny;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo];
    vz_kmo = dev_conserved[3*n_cells + kmo] / dev_conserved[kmo];
    vz_kpo = dev_conserved[3*n_cells + kpo] / dev_conserved[kpo];
    #endif

    // update the conserved variable array
    dev_conserved_half[            id] = dev_conserved[            id]
                                       + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id])
                                       + dtodz * (dev_F_z[            kmo] - dev_F_z[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id]
                                       + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                                       + dtodz * (dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id]
                                       + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                                       + dtodz * (dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id]
                                       + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                                       + dtodz * (dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id]
                                       + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                                       + dtodz * (dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved_half[(5+i)*n_cells + id] = dev_conserved[(5+i)*n_cells + id]
                                         + dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                         + dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                         + dtodz * (dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved_half[(n_fields-1)*n_cells + id] = dev_conserved[(n_fields-1)*n_cells + id]
                                       + dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                       + dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                       + dtodz * (dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id])
                                       + 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
    #endif

    #ifdef DENSITY_FLOOR
    if ( dev_conserved_half[            id] < density_floor ){
      dens_0 = dev_conserved_half[            id];
      printf("###Thread density change  %f -> %f \n", dens_0, density_floor );
      dev_conserved_half[            id] = density_floor;
      // Scale the conserved values to the new density
      dev_conserved_half[1*n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[2*n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[3*n_cells + id] *= (density_floor / dens_0);
      dev_conserved_half[4*n_cells + id] *= (density_floor / dens_0);
      #ifdef DE
      dev_conserved_half[(n_fields-1)*n_cells + id] *= (density_floor / dens_0);
      #endif
    }
    #endif
    //if (dev_conserved_half[id] < 0.0 || dev_conserved_half[id] != dev_conserved_half[id] || dev_conserved_half[4*n_cells+id] < 0.0 || dev_conserved_half[4*n_cells+id] != dev_conserved_half[4*n_cells+id]) {
      //printf("%3d %3d %3d Thread crashed in half step update. d: %e E: %e\n", xid, yid, zid, dev_conserved_half[id], dev_conserved_half[4*n_cells+id]);
    //}

  }

}




#endif //VL
#endif //CUDA
