/*! \file VL_2D_cuda.cu
 *  \brief Definitions of the cuda 2D VL algorithm functions. */

#ifdef VL

  #include <math.h>
  #include <stdio.h>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../hydro/hydro_cuda.h"
  #include "../integrators/VL_2D_cuda.h"
  #include "../reconstruction/pcm_cuda.h"
  #include "../reconstruction/plmc_cuda.h"
  #include "../reconstruction/plmp_cuda.h"
  #include "../reconstruction/ppmc_cuda.h"
  #include "../reconstruction/ppmp_cuda.h"
  #include "../riemann_solvers/exact_cuda.h"
  #include "../riemann_solvers/hllc_cuda.h"
  #include "../riemann_solvers/roe_cuda.h"
  #include "../utils/gpu.hpp"

__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x,
                                                   Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy,
                                                   Real dt, Real gamma, int n_fields);

void VL_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy,
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
    // allocate GPU arrays
    // GPU_Error_Check( cudaMalloc((void**)&dev_conserved,
    // n_fields*n_cells*sizeof(Real)) );
    dev_conserved = d_conserved;
    GPU_Error_Check(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
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

  // Step 1: Use PCM reconstruction to put conserved variables into interface
  // arrays
  hipLaunchKernelGGL(PCM_Reconstruction_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx, ny,
                     n_ghost, gama, n_fields);
  GPU_Error_Check();

  // Step 2: Calculate first-order upwind fluxes
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

  // Step 3: Update the conserved variables half a timestep
  hipLaunchKernelGGL(Update_Conserved_Variables_2D_half, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, dev_conserved_half,
                     F_x, F_y, nx, ny, n_ghost, dx, dy, 0.5 * dt, gama, n_fields);
  GPU_Error_Check();

  // Step 4: Construct left and right interface values using updated conserved
  // variables
  #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
                     dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
                     dt, gama, 1, n_fields);
  #endif
  #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda<0>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, dx, dt,
                     gama, n_fields);
  hipLaunchKernelGGL(PLMC_cuda<1>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, dy, dt,
                     gama, n_fields);
  #endif
  #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx,
                     dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy,
                     dt, gama, 1, n_fields);
  #endif  // PPMP
  #ifdef PPMC
  hipLaunchKernelGGL(PPMC_VL<0>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, gama);
  hipLaunchKernelGGL(PPMC_VL<1>, dim2dGrid, dim1dBlock, 0, 0, dev_conserved_half, Q_Ly, Q_Ry, nx, ny, nz, gama);
  #endif  // PPMC
  GPU_Error_Check();

  // Step 5: Calculate the fluxes again
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
  // Compute the divergence of velocity before updating the conserved array,
  // this solves synchronization issues when adding this term on
  // Update_Conserved_Variables
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx,
                     Q_Ly, Q_Ry, nx, ny, n_ghost, dx, dy, dt, gama, n_fields);
  #endif

  // Step 6: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, F_x, F_y, nx, ny, x_off,
                     y_off, n_ghost, dx, dy, xbound, ybound, dt, gama, n_fields, custom_grav);
  GPU_Error_Check();

  #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, gama, n_fields);
  GPU_Error_Check();
  #endif

  return;
}

void Free_Memory_VL_2D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(F_x);
  cudaFree(F_y);
}

__global__ void Update_Conserved_Variables_2D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x,
                                                   Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy,
                                                   Real dt, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  Real dtodx = dt / dx;
  Real dtody = dt / dy;

  n_cells = nx * ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  id          = threadIdx.x + blockId * blockDim.x;
  yid         = id / nx;
  xid         = id - yid * nx;

  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, P;
  int ipo, jpo;
  #endif

  // all threads but one outer ring of ghost cells
  if (xid > 0 && xid < nx - 1 && yid > 0 && yid < ny - 1) {
    imo = xid - 1 + yid * nx;
    jmo = xid + (yid - 1) * nx;
  #ifdef DE
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    P     = (dev_conserved[4 * n_cells + id] - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
    // if (d < 0.0 || d != d) printf("Negative density before half step
    // update.\n"); if (P < 0.0) printf("%d Negative pressure before half step
    // update.\n", id);
    ipo    = xid + 1 + yid * nx;
    jpo    = xid + (yid + 1) * nx;
    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2 * n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2 * n_cells + jpo] / dev_conserved[jpo];
  #endif
    // update the conserved variable array
    dev_conserved_half[id] =
        dev_conserved[id] + dtodx * (dev_F_x[imo] - dev_F_x[id]) + dtody * (dev_F_y[jmo] - dev_F_y[id]);
    dev_conserved_half[n_cells + id] = dev_conserved[n_cells + id] +
                                       dtodx * (dev_F_x[n_cells + imo] - dev_F_x[n_cells + id]) +
                                       dtody * (dev_F_y[n_cells + jmo] - dev_F_y[n_cells + id]);
    dev_conserved_half[2 * n_cells + id] = dev_conserved[2 * n_cells + id] +
                                           dtodx * (dev_F_x[2 * n_cells + imo] - dev_F_x[2 * n_cells + id]) +
                                           dtody * (dev_F_y[2 * n_cells + jmo] - dev_F_y[2 * n_cells + id]);
    dev_conserved_half[3 * n_cells + id] = dev_conserved[3 * n_cells + id] +
                                           dtodx * (dev_F_x[3 * n_cells + imo] - dev_F_x[3 * n_cells + id]) +
                                           dtody * (dev_F_y[3 * n_cells + jmo] - dev_F_y[3 * n_cells + id]);
    dev_conserved_half[4 * n_cells + id] = dev_conserved[4 * n_cells + id] +
                                           dtodx * (dev_F_x[4 * n_cells + imo] - dev_F_x[4 * n_cells + id]) +
                                           dtody * (dev_F_y[4 * n_cells + jmo] - dev_F_y[4 * n_cells + id]);
  #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved_half[(5 + i) * n_cells + id] =
          dev_conserved[(5 + i) * n_cells + id] +
          dtodx * (dev_F_x[(5 + i) * n_cells + imo] - dev_F_x[(5 + i) * n_cells + id]) +
          dtody * (dev_F_y[(5 + i) * n_cells + jmo] - dev_F_y[(5 + i) * n_cells + id]);
    }
  #endif
  #ifdef DE
    dev_conserved_half[(n_fields - 1) * n_cells + id] =
        dev_conserved[(n_fields - 1) * n_cells + id] +
        dtodx * (dev_F_x[(n_fields - 1) * n_cells + imo] - dev_F_x[(n_fields - 1) * n_cells + id]) +
        dtody * (dev_F_y[(n_fields - 1) * n_cells + jmo] - dev_F_y[(n_fields - 1) * n_cells + id]) +
        0.5 * P * (dtodx * (vx_imo - vx_ipo) + dtody * (vy_jmo - vy_jpo));
  #endif
  }
}

#endif  // VL
