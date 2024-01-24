/*! \file simple_3D_cuda.cu
 *  \brief Definitions of the cuda 3D simple algorithm functions. */

#ifdef CUDA
  #ifdef SIMPLE

    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../hydro/hydro_cuda.h"
    #include "../integrators/simple_3D_cuda.h"
    #include "../io/io.h"
    #include "../reconstruction/pcm_cuda.h"
    #include "../reconstruction/plmc_cuda.h"
    #include "../reconstruction/plmp_cuda.h"
    #include "../reconstruction/ppmc_cuda.h"
    #include "../reconstruction/ppmp_cuda.h"
    #include "../riemann_solvers/exact_cuda.h"
    #include "../riemann_solvers/hll_cuda.h"
    #include "../riemann_solvers/hllc_cuda.h"
    #include "../riemann_solvers/roe_cuda.h"
    #include "../utils/gpu.hpp"

void Simple_Algorithm_3D_CUDA(Real *d_conserved, Real *d_grav_potential, int nx, int ny, int nz, int x_off, int y_off,
                              int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound,
                              Real dt, int n_fields, int custom_grav, Real density_floor,
                              Real *host_grav_potential)
{
  // Here, *dev_conserved contains the entire
  // set of conserved variables on the grid
  // concatenated into a 1-d array
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  // host_grav_potential is NULL if not using GRAVITY
  temp_potential = host_grav_potential;

  if (!memory_allocated) {
    size_t global_free, global_total;
    GPU_Error_Check(cudaMemGetInfo(&global_free, &global_total));

    // allocate memory on the GPU
    chprintf(
        " Allocating Hydro Memory: nfields: %d   n_cells: %d   nx: %d  ny: %d  "
        "nz: %d \n",
        n_fields, n_cells, nx, ny, nz);
    chprintf(" Memory needed: %f GB    Free: %f GB    Total:  %f GB  \n", n_fields * n_cells * sizeof(Real) / 1e9,
             global_free / 1e9, global_total / 1e9);
    dev_conserved = d_conserved;
    GPU_Error_Check(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Ly, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Ry, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Lz, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&Q_Rz, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_y, n_fields * n_cells * sizeof(Real)));
    GPU_Error_Check(cudaMalloc((void **)&F_z, n_fields * n_cells * sizeof(Real)));

    #if defined(GRAVITY)
    // GPU_Error_Check( cudaMalloc((void**)&dev_grav_potential,
    // n_cells*sizeof(Real)) );
    dev_grav_potential = d_grav_potential;
    #else
    dev_grav_potential = NULL;
    #endif

    // If memory is single allocated: memory_allocated becomes true and
    // successive timesteps won't allocate memory. If the memory is not single
    // allocated: memory_allocated remains Null and memory is allocated every
    // timestep.
    memory_allocated = true;
    chprintf(" Memory allocated \n");
  }

    #if defined(GRAVITY) && !defined(GRAVITY_GPU)
  GPU_Error_Check(cudaMemcpy(dev_grav_potential, temp_potential, n_cells * sizeof(Real), cudaMemcpyHostToDevice));
    #endif

    // Step 1: Construct left and right interface values using updated conserved
    // variables
    #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz,
                     Q_Rz, nx, ny, nz, n_ghost, gama, n_fields);
    #endif
    #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt,
                     gama, 2, n_fields);
    #endif  // PLMP
    #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
                     n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama, 1,
                     n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, dz, dt, gama, 2,
                     n_fields);
    #endif
    #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt,
                     gama, 2, n_fields);
    #endif  // PPMP
    #ifdef PPMC
  hipLaunchKernelGGL(PPMC_CTU, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0);
  hipLaunchKernelGGL(PPMC_CTU, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, dy, dt, gama, 1);
  hipLaunchKernelGGL(PPMC_CTU, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, dz, dt, gama, 2);
  GPU_Error_Check();
    #endif  // PPMC

    // Step 2: Calculate the fluxes
    #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
    #endif  // EXACT
    #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
    #endif  // ROE
    #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost,
                     gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost,
                     gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost,
                     gama, 2, n_fields);
    #endif  // HLLC
    #ifdef HLL
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama,
                     1, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama,
                     2, n_fields);
    #endif  // HLL
  GPU_Error_Check();

    #ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this
  // solves synchronization issues when adding this term on
  // Update_Conserved_Variables_3D
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx,
                     Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dx, dy, dz, dt, gama, n_fields);
  GPU_Error_Check();
    #endif

  // Step 3: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry,
                     Q_Lz, Q_Rz, F_x, F_y, F_z, nx, ny, nz, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound,
                     zbound, dt, gama, n_fields, custom_grav, density_floor, dev_grav_potential);
  GPU_Error_Check();

    #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost,
                     n_fields);
  hipLaunchKernelGGL(Sync_Energies_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, gama, n_fields);
  GPU_Error_Check();
    #endif

  return;
}

void Free_Memory_Simple_3D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(Q_Lz);
  cudaFree(Q_Rz);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(F_z);
}

  #endif  // SIMPLE
#endif    // CUDA
