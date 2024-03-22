/*! \file simple_2D_cuda.cu
 *  \brief Definitions of the cuda 2D simple algorithm functions. */

#include <math.h>
#include <stdio.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/simple_2D_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../utils/gpu.hpp"

void Simple_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
                              Real xbound, Real ybound, Real dt, int n_fields, int custom_grav)
{
  // Here, *dev_conserved contains the entire
  // set of conserved variables on the grid
  // concatenated into a 1-d array
  int n_cells             = nx * ny;
  [[maybe_unused]] int nz = 1;
  int ngrid               = (n_cells + TPB - 1) / TPB;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim2dGrid(ngrid, 1, 1);
  // number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  if (!memory_allocated) {
    // allocate memory on the GPU
    dev_conserved = d_conserved;
    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
    // n_fields*n_cells*sizeof(Real)) );
    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_y, n_fields * n_cells * sizeof(Real)));

    // If memory is single allocated: memory_allocated becomes true and
    // successive timesteps won't allocate memory. If the memory is not single
    // allocated: memory_allocated remains Null and memory is allocated every
    // timestep.
    memory_allocated = true;
  }

// Step 1: Do the reconstruction
#ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx, ny,
                     n_ghost, gama, n_fields);
#endif
#ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
                     gama, 1, n_fields);
#endif
#ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda<0>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama,
                     n_fields);
  hipLaunchKernelGGL(PLMC_cuda<1>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama,
                     n_fields);
#endif
#ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
                     gama, 1, n_fields);
#endif
#ifdef PPMC
  hipLaunchKernelGGL(PPMC_CTU<0>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama);
  hipLaunchKernelGGL(PPMC_CTU<1>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama);
#endif
  GPU_Error_Check();

// Step 2: Calculate the fluxes
#ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
#endif
#ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
#endif
#ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
#endif
  GPU_Error_Check();

#ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this
  // solves synchronization issues when adding this term on
  // Update_Conserved_Variables
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx,
                     Q_Ly, Q_Ry, nx, ny, n_ghost, dx, dy, dt, gama, n_fields);
#endif

  // Step 3: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, F_x, F_y, nx, ny, x_off,
                     y_off, n_ghost, dx, dy, xbound, ybound, dt, gama, n_fields, custom_grav);
  GPU_Error_Check();

// Synchronize the total and internal energy
#ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, gama, n_fields);
  GPU_Error_Check();
#endif

  return;
}

void Free_Memory_Simple_2D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(F_x);
  cudaFree(F_y);
}
