/*! \file VL_2D_cuda.cu
 *  \brief Definitions of the cuda 2D VL algorithm functions. */

#ifdef CUDA
#ifdef VL

#include <stdio.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/VL_2D_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../old_cholla/h_correction_2D_cuda.h"
#include "../cooling/cooling_cuda.h"
#include "../old_cholla/subgrid_routines_2D.h"


__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half,
                                                   Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                                   int n_ghost, Real dx, Real dy, Real dt, Real gamma, int n_fields);


Real VL_Algorithm_2D_CUDA ( Real *host_conserved0, Real *host_conserved1,
   Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost,
   Real dx, Real dy, Real xbound, Real ybound, Real dt, int n_fields)
{

  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  //host_conserved0 contains the values at time n,
  //host_conserved1 will contain the values at time n+1

  // Initialize dt values
  Real max_dti = 0;
  #ifdef COOLING_GPU
  Real min_dt = 1e10;
  #endif

  if ( !block_size ) {
    // calculate the dimensions for each subgrid block
    sub_dimensions_2D(nx, ny, n_ghost, &nx_s, &ny_s, &block1_tot, &block2_tot, &remainder1, &remainder2, n_fields);
    //printf("%d %d %d %d %d %d\n", nx_s, ny_s, block1_tot, block2_tot, remainder1, remainder2);
    nz_s = 1;
    block_tot = block1_tot*block2_tot;
    // number of cells in one subgrid block
    BLOCK_VOL = nx_s*ny_s*nz_s;
    // dimensions for the 1D GPU grid
    ngrid = (BLOCK_VOL + TPB - 1) / (TPB);
    #ifndef DYNAMIC_GPU_ALLOC
    block_size = true;
    #endif
  }
  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim2dGrid(ngrid, 1, 1);
  //number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  // Set up pointers for the location to copy from and to
  if (block_tot == 1) {
    tmp1 = host_conserved0;
    tmp2 = host_conserved1;
  }

  if ( !memory_allocated ) {

    // allocate buffer to copy conserved variable blocks from and to
    if (block_tot > 1) {
      if ( cudaSuccess != cudaHostAlloc(&buffer, n_fields*BLOCK_VOL*sizeof(Real), cudaHostAllocDefault) ) {
        printf("Failed to allocate CPU buffer.\n");
      }
      tmp1 = buffer;
      tmp2 = buffer;
    }
    // allocate an array on the CPU to hold max_dti returned from each thread block
    CudaSafeCall( cudaHostAlloc(&host_dti_array, ngrid*sizeof(Real), cudaHostAllocDefault) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaHostAlloc(&host_dt_array, ngrid*sizeof(Real), cudaHostAllocDefault) );
    #endif

    // allocate GPU arrays
    //CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
    dev_conserved = d_conserved;
    CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif

    #ifndef DYNAMIC_GPU_ALLOC
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif
    //d_conserved = dev_conserved;
  }


  // counter for which block we're on
  int block = 0;


  // START LOOP OVER SUBGRID BLOCKS HERE
  while (block < block_tot) {

    // copy the conserved variable block to the buffer
    host_copy_block_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved0, buffer, n_fields);

    // calculate the global x and y offsets of this subgrid block
    // (only needed for gravitational potential)
    get_offsets_2D(nx_s, ny_s, n_ghost, x_off, y_off, block, block1_tot, block2_tot, remainder1, remainder2, &x_off_s, &y_off_s);

    // copy the conserved variables onto the GPU
    #ifndef HYDRO_GPU
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #endif

    // Step 1: Use PCM reconstruction to put conserved variables into interface arrays
    hipLaunchKernelGGL(PCM_Reconstruction_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx_s, ny_s, n_ghost, gama, n_fields);
    CudaCheckError();


    // Step 2: Calculate first-order upwind fluxes
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    CudaCheckError();


    // Step 3: Update the conserved variables half a timestep
    hipLaunchKernelGGL(Update_Conserved_Variables_2D_half, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved_half, F_x, F_y, nx_s, ny_s, n_ghost, dx, dy, 0.5*dt, gama, n_fields);
    CudaCheckError();


    // Step 4: Construct left and right interface values using updated conserved variables
    #ifdef PLMP
    hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    #ifdef PLMC
    hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    #ifdef PPMP
    hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif //PPMP
    #ifdef PPMC
    hipLaunchKernelGGL(PPMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif //PPMC
    CudaCheckError();


    // Step 5: Calculate the fluxes again
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    CudaCheckError();

    #ifdef DE
    // Compute the divergence of velocity before updating the conserved array, this solves syncronization issues when adding this term on Update_Conserved_Variables
    hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0,  dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx_s, ny_s, n_ghost, dx, dy, dt, gama, n_fields );
    #endif


    // Step 6: Update the conserved variable array
    hipLaunchKernelGGL(Update_Conserved_Variables_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, F_x, F_y, nx_s, ny_s, x_off_s, y_off_s, n_ghost, dx, dy, xbound, ybound, dt, gama, n_fields);
    CudaCheckError();


    #ifdef DE
    hipLaunchKernelGGL(Select_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, n_ghost, n_fields);
    hipLaunchKernelGGL(Sync_Energies_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #endif

    /*
    // Apply cooling
    #ifdef COOLING_GPU
    hipLaunchKernelGGL(cooling_kernel, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, dt, gama, dev_dt_array);
    CudaCheckError();
    #endif


    // copy the conserved variable array back to the CPU
    #ifndef HYDRO_GPU
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
    #endif

    // copy the updated conserved variable array back into the host_conserved array on the CPU
    host_return_block_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved1, buffer, n_fields);

    #ifdef COOLING_GPU
    // copy the dt array from cooling onto the CPU
    CudaSafeCall( cudaMemcpy(host_dt_array, dev_dt_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // iterate through to find the minimum dt for this subgrid block
    for (int i=0; i<ngrid; i++) {
      min_dt = fmin(min_dt, host_dt_array[i]);
    }
    if (min_dt < C_cfl/max_dti) {
      max_dti = C_cfl/min_dt;
    }
    #endif
    */
    // add one to the counter
    block++;

  }

  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_VL_2D();
  #endif


  // return the maximum inverse timestep
  return max_dti;

}

void Free_Memory_VL_2D() {

  // free the CPU memory
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
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif

}


__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;

  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;

  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, P;
  int ipo, jpo;
  #endif


  // all threads but one outer ring of ghost cells
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1)
  {
    imo = xid-1 + yid*nx;
    jmo = xid + (yid-1)*nx;
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before half step update.\n");
    //if (P < 0.0) printf("%d Negative pressure before half step update.\n", id);
    ipo = xid+1 + yid*nx;
    jpo = xid + (yid+1)*nx;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo];
    #endif
    // update the conserved variable array
    dev_conserved_half[            id] = dev_conserved[            id]
                                       + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id]
                                       + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id]
                                       + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id]
                                       + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id]
                                       + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved_half[(5+i)*n_cells + id] = dev_conserved[(5+i)*n_cells + id]
                                         + dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                         + dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved_half[(n_fields-1)*n_cells + id] = dev_conserved[(n_fields-1)*n_cells + id]
                                       + dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                       + dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                       + 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo));
    #endif
  }
}




#endif //VL
#endif //CUDA

