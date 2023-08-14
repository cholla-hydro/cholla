/*! \file VL_1D_cuda.cu
 *  \brief Definitions of the cuda VL algorithm functions. */

#ifdef CUDA
  #ifdef VL

    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>

    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../hydro/hydro_cuda.h"
    #include "../integrators/VL_1D_cuda.h"
    #include "../io/io.h"
    #include "../reconstruction/pcm_cuda.h"
    #include "../reconstruction/plmc_cuda.h"
    #include "../reconstruction/plmp_cuda.h"
    #include "../reconstruction/ppmc_cuda.h"
    #include "../reconstruction/ppmp_cuda.h"
    #include "../riemann_solvers/exact_cuda.h"
    #include "../riemann_solvers/hllc_cuda.h"
    #include "../riemann_solvers/roe_cuda.h"
    #include "../utils/error_handling.h"
    #include "../utils/gpu.hpp"

__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F,
                                                   int n_cells, int n_ghost, Real dx, Real dt, Real gamma,
                                                   int n_fields);

void VL_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
                          int n_fields)
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
    // CudaSafeCall( cudaMalloc((void**)&dev_conserved,
    // n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall(cudaMalloc((void **)&dev_conserved_half, n_fields * n_cells * sizeof(Real)));
    CudaSafeCall(cudaMalloc((void **)&Q_Lx, n_fields * n_cells * sizeof(Real)));
    CudaSafeCall(cudaMalloc((void **)&Q_Rx, n_fields * n_cells * sizeof(Real)));
    CudaSafeCall(cudaMalloc((void **)&F_x, n_fields * n_cells * sizeof(Real)));

    // If memory is single allocated: memory_allocated becomes true and
    // successive timesteps won't allocate memory. If the memory is not single
    // allocated: memory_allocated remains Null and memory is allocated every
    // timestep.
    memory_allocated = true;
  }

  // Step 1: Use PCM reconstruction to put conserved variables into interface
  // arrays
  hipLaunchKernelGGL(PCM_Reconstruction_1D, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, n_ghost, gama,
                     n_fields);
  CudaCheckError();

    // Step 2: Calculate first-order upwind fluxes
    #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
    #endif
    #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
                     n_fields);
    #endif
    #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
                     n_fields);
    #endif
  CudaCheckError();

  // Step 3: Update the conserved variables half a timestep
  hipLaunchKernelGGL(Update_Conserved_Variables_1D_half, dimGrid, dimBlock, 0, 0, dev_conserved, dev_conserved_half,
                     F_x, n_cells, n_ghost, dx, 0.5 * dt, gama, n_fields);
  CudaCheckError();

    // Step 4: Construct left and right interface values using updated conserved
    // variables
    #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_1D, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, n_ghost, gama,
                     n_fields);
    #endif
    #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, dx, dt, gama, 0,
                     n_fields);
    #endif
    #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
    #endif
    #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt,
                     gama, 0, n_fields);
    #endif
    #ifdef PPMC
  hipLaunchKernelGGL(PPMC_VL, dimGrid, dimBlock, 0, 0, dev_conserved_half, Q_Lx, Q_Rx, nx, ny, nz, gama, 0);
    #endif
  CudaCheckError();

    // Step 5: Calculate the fluxes again
    #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama,
                     0, n_fields);
    #endif
    #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
                     n_fields);
    #endif
    #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0,
                     n_fields);
    #endif
  CudaCheckError();

    #ifdef DE
  // Compute the divergence of velocity before updating the conserved array,
  // this solves synchronization issues when adding this term on
  // Update_Conserved_Variables
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_1D, dimGrid, dimBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx,
                     n_ghost, dx, dt, gama, n_fields);
    #endif

  // Step 6: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_1D, dimGrid, dimBlock, 0, 0, dev_conserved, F_x, n_cells, x_off,
                     n_ghost, dx, xbound, dt, gama, n_fields);
  CudaCheckError();

    #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_1D, dimGrid, dimBlock, 0, 0, dev_conserved, nx, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_1D, dimGrid, dimBlock, 0, 0, dev_conserved, nx, n_ghost, gama, n_fields);
  CudaCheckError();
    #endif

  return;
}

void Free_Memory_VL_1D()
{
  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(F_x);
}

__global__ void Update_Conserved_Variables_1D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F,
                                                   int n_cells, int n_ghost, Real dx, Real dt, Real gamma, int n_fields)
{
  int id, imo;
  Real dtodx = dt / dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;

    #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, P;
  int ipo;
    #endif

  // threads corresponding all cells except outer ring of ghost cells do the
  // calculation
  if (id > 0 && id < n_cells - 1) {
    imo = id - 1;
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
    ipo    = id + 1;
    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];
    #endif
    // update the conserved variable array
    dev_conserved_half[id] = dev_conserved[id] + dtodx * (dev_F[imo] - dev_F[id]);
    dev_conserved_half[n_cells + id] =
        dev_conserved[n_cells + id] + dtodx * (dev_F[n_cells + imo] - dev_F[n_cells + id]);
    dev_conserved_half[2 * n_cells + id] =
        dev_conserved[2 * n_cells + id] + dtodx * (dev_F[2 * n_cells + imo] - dev_F[2 * n_cells + id]);
    dev_conserved_half[3 * n_cells + id] =
        dev_conserved[3 * n_cells + id] + dtodx * (dev_F[3 * n_cells + imo] - dev_F[3 * n_cells + id]);
    dev_conserved_half[4 * n_cells + id] =
        dev_conserved[4 * n_cells + id] + dtodx * (dev_F[4 * n_cells + imo] - dev_F[4 * n_cells + id]);
    #ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved_half[(5 + i) * n_cells + id] =
          dev_conserved[(5 + i) * n_cells + id] +
          dtodx * (dev_F[(5 + i) * n_cells + imo] - dev_F[(5 + i) * n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_conserved_half[(n_fields - 1) * n_cells + id] =
        dev_conserved[(n_fields - 1) * n_cells + id] +
        dtodx * (dev_F[(n_fields - 1) * n_cells + imo] - dev_F[(n_fields - 1) * n_cells + id]) +
        0.5 * P * (dtodx * (vx_imo - vx_ipo));
    #endif
  }
}

  #endif  // VL
#endif    // CUDA
