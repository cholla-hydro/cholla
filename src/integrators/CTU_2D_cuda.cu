/*! \file CTU_2D_cuda.cu
 *  \brief Definitions of the cuda 2D CTU algorithm functions. */

#ifdef CUDA

#include <stdio.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/CTU_2D_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../old_cholla/h_correction_2D_cuda.h"



__global__ void Evolve_Interface_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F1_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F1_y,
                                           int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, int n_fields);


void CTU_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, int n_fields)
{

  //Here, *dev_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  int n_cells = nx*ny;
  int nz = 1;

  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim2dGrid(ngrid, 1, 1);
  //number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  if ( !memory_allocated ) {

    // allocate memory on the GPU
    dev_conserved = d_conserved;
    //CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,  n_fields*n_cells*sizeof(Real)) );

    // If memory is single allocated: memory_allocated becomes true and successive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
  }

  // Step 1: Do the reconstruction
  #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx, ny, n_ghost, gama, n_fields);
  #endif
  #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  #endif
  #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  #endif
  #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  #endif
  #ifdef PPMC
  hipLaunchKernelGGL(PPMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMC_cuda, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  #endif
  CudaCheckError();


  // Step 2: Calculate the fluxes
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  CudaCheckError();

#ifdef CTU

  // Step 3: Evolve the interface states
  hipLaunchKernelGGL(Evolve_Interface_States_2D, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, nx, ny, n_ghost, dx, dy, dt, n_fields);
  CudaCheckError();


  // Step 4: Calculate the fluxes again
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim2dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  #endif
  CudaCheckError();

#endif //CTU

  #ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this solves synchronization issues when adding this term on Update_Conserved_Variables
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0,  dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx, ny, n_ghost, dx, dy, dt, gama, n_fields );
  #endif


  // Step 5: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, F_x, F_y, nx, ny, x_off, y_off, n_ghost, dx, dy, xbound, ybound, dt, gama, n_fields);
  CudaCheckError();

  // Synchronize the total and internal energy
  #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_2D, dim2dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif

  return;

}

void Free_Memory_CTU_2D() {

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(F_x);
  cudaFree(F_y);

}


__global__ void Evolve_Interface_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, int n_fields)
{
  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  int n_cells = nx*ny;

  // get a thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  int tid = threadIdx.x + blockId * blockDim.x;
  int yid = tid / nx;
  int xid = tid - yid*nx;
  int id = xid + yid*nx;


  // set the new x interface states
  if (xid > n_ghost-2 && xid < nx-n_ghost && yid > n_ghost-2 && yid < ny-n_ghost+1)
  {
    // left
    int ipo = xid+1 + yid*nx;
    int jmo = xid + (yid-1)*nx;
    int ipojmo = xid+1 + (yid-1)*nx;
    dev_Q_Lx[            id] += 0.5*dtody*(dev_F_y[            jmo] - dev_F_y[            id]);
    dev_Q_Lx[  n_cells + id] += 0.5*dtody*(dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_Q_Lx[2*n_cells + id] += 0.5*dtody*(dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]);
    dev_Q_Lx[3*n_cells + id] += 0.5*dtody*(dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_Q_Lx[4*n_cells + id] += 0.5*dtody*(dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Lx[(5+i)*n_cells + id] += 0.5*dtody*(dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_Q_Lx[(n_fields-1)*n_cells + id] += 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id]);
    #endif
    // right
    dev_Q_Rx[            id] += 0.5*dtody*(dev_F_y[            ipojmo] - dev_F_y[            ipo]);
    dev_Q_Rx[  n_cells + id] += 0.5*dtody*(dev_F_y[  n_cells + ipojmo] - dev_F_y[  n_cells + ipo]);
    dev_Q_Rx[2*n_cells + id] += 0.5*dtody*(dev_F_y[2*n_cells + ipojmo] - dev_F_y[2*n_cells + ipo]);
    dev_Q_Rx[3*n_cells + id] += 0.5*dtody*(dev_F_y[3*n_cells + ipojmo] - dev_F_y[3*n_cells + ipo]);
    dev_Q_Rx[4*n_cells + id] += 0.5*dtody*(dev_F_y[4*n_cells + ipojmo] - dev_F_y[4*n_cells + ipo]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Rx[(5+i)*n_cells + id] += 0.5*dtody*(dev_F_y[(5+i)*n_cells + ipojmo] - dev_F_y[(5+i)*n_cells + ipo]);
    }
    #endif
    #ifdef DE
    dev_Q_Rx[(n_fields-1)*n_cells + id] += 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + ipojmo] - dev_F_y[(n_fields-1)*n_cells + ipo]);
    #endif
  }
  // set the new y interface states
  if (yid > n_ghost-2 && yid < ny-n_ghost && xid > n_ghost-2 && xid < nx-n_ghost+1)
  {
    // left
    int jpo = xid + (yid+1)*nx;
    int imo = xid-1 + yid*nx;
    int jpoimo = xid-1 + (yid+1)*nx;
    dev_Q_Ly[            id] += 0.5*dtodx*(dev_F_x[            imo] - dev_F_x[            id]);
    dev_Q_Ly[  n_cells + id] += 0.5*dtodx*(dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]);
    dev_Q_Ly[2*n_cells + id] += 0.5*dtodx*(dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]);
    dev_Q_Ly[3*n_cells + id] += 0.5*dtodx*(dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id]);
    dev_Q_Ly[4*n_cells + id] += 0.5*dtodx*(dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Ly[(5+i)*n_cells + id] += 0.5*dtodx*(dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_Q_Ly[(n_fields-1)*n_cells + id] += 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id]);
    #endif
    // right
    dev_Q_Ry[            id] += 0.5*dtodx*(dev_F_x[            jpoimo] - dev_F_x[            jpo]);
    dev_Q_Ry[  n_cells + id] += 0.5*dtodx*(dev_F_x[  n_cells + jpoimo] - dev_F_x[  n_cells + jpo]);
    dev_Q_Ry[2*n_cells + id] += 0.5*dtodx*(dev_F_x[2*n_cells + jpoimo] - dev_F_x[2*n_cells + jpo]);
    dev_Q_Ry[3*n_cells + id] += 0.5*dtodx*(dev_F_x[3*n_cells + jpoimo] - dev_F_x[3*n_cells + jpo]);
    dev_Q_Ry[4*n_cells + id] += 0.5*dtodx*(dev_F_x[4*n_cells + jpoimo] - dev_F_x[4*n_cells + jpo]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Ry[(5+i)*n_cells + id] += 0.5*dtodx*(dev_F_x[(5+i)*n_cells + jpoimo] - dev_F_x[(5+i)*n_cells + jpo]);
    }
    #endif
    #ifdef DE
    dev_Q_Ry[(n_fields-1)*n_cells + id] += 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + jpoimo] - dev_F_x[(n_fields-1)*n_cells + jpo]);
    #endif
  }

}


#endif //CUDA

