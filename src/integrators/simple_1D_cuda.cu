/*! \file simple_1D_cuda.cu
 *  \brief Definitions of the 1D simple algorithm functions. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/simple_1D_cuda.h"
#include "../io/io.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../reconstruction/reconstruction.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"

void Simple_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
                              int n_fields, int custom_grav)
{
  // Here, *dev_conserved contains the entire
  // set of conserved variables on the grid

  int n_cells             = nx;
  [[maybe_unused]] int ny = 1;
  [[maybe_unused]] int nz = 1;
  int ngrid               = (n_cells + TPB - 1) / TPB;

  // set the dimensions of the cuda grid
  dim3 dimGrid(ngrid, 1, 1);
  dim3 dimBlock(TPB, 1, 1);

  if (!memory_allocated) {
    // allocate memory on the GPU
    dev_conserved = d_conserved;
    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
    // n_fields*n_cells*sizeof(Real)) );
    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_x, (n_fields)*n_cells * sizeof(Real)));

    // If memory is single allocated: memory_allocated becomes true and
    // successive timesteps won't allocate memory. If the memory is not single
    // allocated: memory_allocated remains Null and memory is allocated every
    // timestep.
    memory_allocated = true;
  }

// Step 1: Do the reconstruction
#ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama,
                     0, n_fields);
  GPU_Error_Check();
#endif
#ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda<0>, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama,
                     n_fields);
  GPU_Error_Check();
#endif
#ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama,
                     0, n_fields);
  GPU_Error_Check();
#endif
#ifdef PPMC
  hipLaunchKernelGGL(PPMC_CTU<0>, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama);
  GPU_Error_Check();
#endif

// Step 2: Calculate the fluxes
#ifdef EXACT
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Calculate_Exact_Fluxes_CUDA<reconstruction::Kind::chosen, 0>), dimGrid, dimBlock,
                     0, 0, dev_conserved, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_cells, gama, n_fields);
#endif
#ifdef ROE
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Calculate_Roe_Fluxes_CUDA<reconstruction::Kind::chosen, 0>), dimGrid, dimBlock, 0,
                     0, dev_conserved, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_cells, gama, n_fields);
#endif
#ifdef HLLC
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Calculate_HLLC_Fluxes_CUDA<reconstruction::Kind::chosen, 0>), dimGrid, dimBlock, 0,
                     0, dev_conserved, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_cells, gama, n_fields);
#endif
  GPU_Error_Check();

#ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this
  // solves synchronization issues when adding this term on
  // Update_Conserved_Variables
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_1D, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx,
                     n_ghost, dx, dt, gama, n_fields);
#endif

  // Step 3: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_1D, dimGrid, dimBlock, 0, 0, dev_conserved, F_x, n_cells, x_off,
                     n_ghost, dx, xbound, dt, gama, n_fields, custom_grav);
  GPU_Error_Check();

// Synchronize the total and internal energy, if using dual-energy formalism
#ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_1D, dimGrid, dimBlock, 0, 0, dev_conserved, nx, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_1D, dimGrid, dimBlock, 0, 0, dev_conserved, n_cells, n_ghost, gama, n_fields);
  GPU_Error_Check();
#endif

  return;
}

void Free_Memory_Simple_1D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(F_x);
}
