/*! \file CTU_3D_cuda.cu
 *  \brief Definitions of the cuda 3D CTU algorithm functions. */

#ifdef CUDA

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../hydro/hydro_cuda.h"
#include "../integrators/CTU_3D_cuda.h"
#include "../reconstruction/pcm_cuda.h"
#include "../reconstruction/plmp_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../reconstruction/ppmp_cuda.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../riemann_solvers/exact_cuda.h"
#include "../riemann_solvers/roe_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"
#include "../old_cholla/h_correction_3D_cuda.h"
#include "../io/io.h"


__global__ void Evolve_Interface_States_3D(Real *dev_conserved, Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           Real *dev_Q_Lz, Real *dev_Q_Rz, Real *dev_F_z,
                                           int nx, int ny, int nz, int n_ghost,
                                           Real dx, Real dy, Real dz, Real dt, int n_fields);


void CTU_Algorithm_3D_CUDA(Real *d_conserved, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, int n_fields , Real density_floor, Real U_floor, Real *host_grav_potential )
{
  //Here, *dev_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array

  int n_cells = nx*ny*nz;
  int ngrid = (n_cells + TPB - 1) / TPB;
  // set values for GPU kernels
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  //host_grav_potential is NULL if not using GRAVITY
  temp_potential = host_grav_potential;

  if ( !memory_allocated ) {

    // allocate memory on the GPU
    dev_conserved = d_conserved;
    //CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lz,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rz,  n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,   n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_z,   n_fields*n_cells*sizeof(Real)) );

    #if defined( GRAVITY )
    CudaSafeCall( cudaMalloc((void**)&dev_grav_potential, n_cells*sizeof(Real)) );
    #else
    dev_grav_potential = NULL;
    #endif

    // If memory is single allocated: memory_allocated becomes true and successive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
  }

  #if defined( GRAVITY ) && !defined ( GRAVITY_GPU )
  CudaSafeCall( cudaMemcpy(dev_grav_potential, temp_potential, n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  #endif


  // Step 1: Do the reconstruction
  #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, gama, n_fields);
  #endif //PCM
  #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PLMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt, gama, 2, n_fields);
  #endif //PLMP
  #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PLMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt, gama, 2, n_fields);
  #endif //PLMC
  #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PPMP_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt, gama, 2, n_fields);
  #endif //PPMP
  #ifdef PPMC
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Ly, Q_Ry, nx, ny, nz, n_ghost, dy, dt, gama, 1, n_fields);
  hipLaunchKernelGGL(PPMC_cuda, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dz, dt, gama, 2, n_fields);
  #endif //PPMC
  CudaCheckError();


  // Step 2: Calculate the fluxes
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //EXACT
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //ROE
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //HLLC
  #ifdef HLL
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //HLLC
  CudaCheckError();


  #ifdef CTU
  // Step 3: Evolve the interface states
  hipLaunchKernelGGL(Evolve_Interface_States_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, dx, dy, dz, dt, n_fields);
  CudaCheckError();


  // Step 4: Calculate the fluxes again
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //EXACT
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //ROE
  #ifdef HLLC
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //HLLC
  #ifdef HLL
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Ly, Q_Ry, F_y, nx, ny, nz, n_ghost, gama, 1, n_fields);
  hipLaunchKernelGGL(Calculate_HLL_Fluxes_CUDA, dim1dGrid, dim1dBlock, 0, 0, Q_Lz, Q_Rz, F_z, nx, ny, nz, n_ghost, gama, 2, n_fields);
  #endif //HLLC
  CudaCheckError();
  #endif //CTU

  #ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this solves synchronization issues when adding this term on Update_Conserved_Variables_3D
  hipLaunchKernelGGL(Partial_Update_Advected_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0,  dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx, ny, nz, n_ghost, dx, dy, dz, dt, gama, n_fields );
  CudaCheckError();
  #endif


  // Step 5: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, F_x, F_y, F_z, nx, ny, nz, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound, zbound, dt, gama, n_fields, density_floor, dev_grav_potential );
  CudaCheckError();


  // Synchronize the total and internal energies
  #ifdef DE
  hipLaunchKernelGGL(Select_Internal_Energy_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields);
  hipLaunchKernelGGL(Sync_Energies_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif //DE

  #ifdef TEMPERATURE_FLOOR
  hipLaunchKernelGGL(Apply_Temperature_Floor, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, U_floor );
  CudaCheckError();
  #endif //TEMPERATURE_FLOOR

  return;

}


void Free_Memory_CTU_3D() {

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
  #if defined( GRAVITY )
  cudaFree(dev_grav_potential);
  #endif

}


__global__ void Evolve_Interface_States_3D(Real *dev_conserved, Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           Real *dev_Q_Lz, Real *dev_Q_Rz, Real *dev_F_z,
                                           int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, int n_fields)
{
  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  int n_cells = nx*ny*nz;

  // get a thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  int id = xid + yid*nx + zid*nx*ny;

  if (xid > n_ghost-3 && xid < nx-n_ghost+1 && yid > n_ghost-2 && yid < ny-n_ghost+1 && zid > n_ghost-2 && zid < nz-n_ghost+1)
  {
    // set the new x interface states
    // left
    int ipo = xid+1 + yid*nx + zid*nx*ny;
    int jmo = xid + (yid-1)*nx + zid*nx*ny;
    int kmo = xid + yid*nx + (zid-1)*nx*ny;
    int ipojmo = xid+1 + (yid-1)*nx + zid*nx*ny;
    int ipokmo = xid+1 + yid*nx + (zid-1)*nx*ny;
    dev_Q_Lx[            id] += 0.5*dtody*(dev_F_y[            jmo] - dev_F_y[            id])
                              + 0.5*dtodz*(dev_F_z[            kmo] - dev_F_z[            id]);
    dev_Q_Lx[  n_cells + id] += 0.5*dtody*(dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                              + 0.5*dtodz*(dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_Q_Lx[2*n_cells + id] += 0.5*dtody*(dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                              + 0.5*dtodz*(dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_Q_Lx[3*n_cells + id] += 0.5*dtody*(dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                              + 0.5*dtodz*(dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_Q_Lx[4*n_cells + id] += 0.5*dtody*(dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                              + 0.5*dtodz*(dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Lx[(5+i)*n_cells + id] += 0.5*dtody*(dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                + 0.5*dtodz*(dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_Q_Lx[(n_fields-1)*n_cells + id] += 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                              + 0.5*dtodz*(dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id]);
    #endif

    // right
    dev_Q_Rx[            id] += 0.5*dtody*(dev_F_y[            ipojmo] - dev_F_y[            ipo])
                              + 0.5*dtodz*(dev_F_z[            ipokmo] - dev_F_z[            ipo]);
    dev_Q_Rx[  n_cells + id] += 0.5*dtody*(dev_F_y[  n_cells + ipojmo] - dev_F_y[  n_cells + ipo])
                              + 0.5*dtodz*(dev_F_z[  n_cells + ipokmo] - dev_F_z[  n_cells + ipo]);
    dev_Q_Rx[2*n_cells + id] += 0.5*dtody*(dev_F_y[2*n_cells + ipojmo] - dev_F_y[2*n_cells + ipo])
                              + 0.5*dtodz*(dev_F_z[2*n_cells + ipokmo] - dev_F_z[2*n_cells + ipo]);
    dev_Q_Rx[3*n_cells + id] += 0.5*dtody*(dev_F_y[3*n_cells + ipojmo] - dev_F_y[3*n_cells + ipo])
                              + 0.5*dtodz*(dev_F_z[3*n_cells + ipokmo] - dev_F_z[3*n_cells + ipo]);
    dev_Q_Rx[4*n_cells + id] += 0.5*dtody*(dev_F_y[4*n_cells + ipojmo] - dev_F_y[4*n_cells + ipo])
                              + 0.5*dtodz*(dev_F_z[4*n_cells + ipokmo] - dev_F_z[4*n_cells + ipo]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Rx[(5+i)*n_cells + id] += 0.5*dtody*(dev_F_y[(5+i)*n_cells + ipojmo] - dev_F_y[(5+i)*n_cells + ipo])
                                + 0.5*dtodz*(dev_F_z[(5+i)*n_cells + ipokmo] - dev_F_z[(5+i)*n_cells + ipo]);
    }
    #endif
    #ifdef DE
    dev_Q_Rx[(n_fields-1)*n_cells + id] += 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + ipojmo] - dev_F_y[(n_fields-1)*n_cells + ipo])
                              + 0.5*dtodz*(dev_F_z[(n_fields-1)*n_cells + ipokmo] - dev_F_z[(n_fields-1)*n_cells + ipo]);
    #endif
  }
  if (yid > n_ghost-3 && yid < ny-n_ghost+1 && xid > n_ghost-2 && xid < nx-n_ghost+1 && zid > n_ghost-2 && zid < nz-n_ghost+1)
  {
    // set the new y interface states
    // left
    int jpo = xid + (yid+1)*nx + zid*nx*ny;
    int imo = xid-1 + yid*nx + zid*nx*ny;
    int kmo = xid + yid*nx + (zid-1)*nx*ny;
    int jpoimo = xid-1 + (yid+1)*nx + zid*nx*ny;
    int jpokmo = xid + (yid+1)*nx + (zid-1)*nx*ny;
    dev_Q_Ly[            id] += 0.5*dtodz*(dev_F_z[            kmo] - dev_F_z[            id])
                              + 0.5*dtodx*(dev_F_x[            imo] - dev_F_x[            id]);
    dev_Q_Ly[  n_cells + id] += 0.5*dtodz*(dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id])
                              + 0.5*dtodx*(dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]);
    dev_Q_Ly[2*n_cells + id] += 0.5*dtodz*(dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id])
                              + 0.5*dtodx*(dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]);
    dev_Q_Ly[3*n_cells + id] += 0.5*dtodz*(dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id])
                              + 0.5*dtodx*(dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id]);
    dev_Q_Ly[4*n_cells + id] += 0.5*dtodz*(dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id])
                              + 0.5*dtodx*(dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Ly[(5+i)*n_cells + id] += 0.5*dtodz*(dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id])
                                + 0.5*dtodx*(dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_Q_Ly[(n_fields-1)*n_cells + id] += 0.5*dtodz*(dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id])
                              + 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id]);
    #endif

    // right
    dev_Q_Ry[            id] += 0.5*dtodz*(dev_F_z[            jpokmo] - dev_F_z[            jpo])
                              + 0.5*dtodx*(dev_F_x[            jpoimo] - dev_F_x[            jpo]);
    dev_Q_Ry[  n_cells + id] += 0.5*dtodz*(dev_F_z[  n_cells + jpokmo] - dev_F_z[  n_cells + jpo])
                              + 0.5*dtodx*(dev_F_x[  n_cells + jpoimo] - dev_F_x[  n_cells + jpo]);
    dev_Q_Ry[2*n_cells + id] += 0.5*dtodz*(dev_F_z[2*n_cells + jpokmo] - dev_F_z[2*n_cells + jpo])
                              + 0.5*dtodx*(dev_F_x[2*n_cells + jpoimo] - dev_F_x[2*n_cells + jpo]);
    dev_Q_Ry[3*n_cells + id] += 0.5*dtodz*(dev_F_z[3*n_cells + jpokmo] - dev_F_z[3*n_cells + jpo])
                              + 0.5*dtodx*(dev_F_x[3*n_cells + jpoimo] - dev_F_x[3*n_cells + jpo]);
    dev_Q_Ry[4*n_cells + id] += 0.5*dtodz*(dev_F_z[4*n_cells + jpokmo] - dev_F_z[4*n_cells + jpo])
                              + 0.5*dtodx*(dev_F_x[4*n_cells + jpoimo] - dev_F_x[4*n_cells + jpo]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Ry[(5+i)*n_cells + id] += 0.5*dtodz*(dev_F_z[(5+i)*n_cells + jpokmo] - dev_F_z[(5+i)*n_cells + jpo])
                                + 0.5*dtodx*(dev_F_x[(5+i)*n_cells + jpoimo] - dev_F_x[(5+i)*n_cells + jpo]);
    }
    #endif
    #ifdef DE
    dev_Q_Ry[(n_fields-1)*n_cells + id] += 0.5*dtodz*(dev_F_z[(n_fields-1)*n_cells + jpokmo] - dev_F_z[(n_fields-1)*n_cells + jpo])
                              + 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + jpoimo] - dev_F_x[(n_fields-1)*n_cells + jpo]);
    #endif
  }
  if (zid > n_ghost-3 && zid < nz-n_ghost+1 && xid > n_ghost-2 && xid < nx-n_ghost+1 && yid > n_ghost-2 && yid < ny-n_ghost+1)
  {
    // set the new z interface states
    // left
    int kpo = xid + yid*nx + (zid+1)*nx*ny;
    int imo = xid-1 + yid*nx + zid*nx*ny;
    int jmo = xid + (yid-1)*nx + zid*nx*ny;
    int kpoimo = xid-1 + yid*nx + (zid+1)*nx*ny;
    int kpojmo = xid + (yid-1)*nx + (zid+1)*nx*ny;
    dev_Q_Lz[            id] += 0.5*dtodx*(dev_F_x[            imo] - dev_F_x[            id])
                              + 0.5*dtody*(dev_F_y[            jmo] - dev_F_y[            id]);
    dev_Q_Lz[  n_cells + id] += 0.5*dtodx*(dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                              + 0.5*dtody*(dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_Q_Lz[2*n_cells + id] += 0.5*dtodx*(dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                              + 0.5*dtody*(dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]);
    dev_Q_Lz[3*n_cells + id] += 0.5*dtodx*(dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                              + 0.5*dtody*(dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_Q_Lz[4*n_cells + id] += 0.5*dtodx*(dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                              + 0.5*dtody*(dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Lz[(5+i)*n_cells + id] += 0.5*dtodx*(dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                + 0.5*dtody*(dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id]);
    }
    #endif
    #ifdef DE
    dev_Q_Lz[(n_fields-1)*n_cells + id] += 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                              + 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id]);
    #endif
    // right
    dev_Q_Rz[            id] += 0.5*dtodx*(dev_F_x[            kpoimo] - dev_F_x[            kpo])
                              + 0.5*dtody*(dev_F_y[            kpojmo] - dev_F_y[            kpo]);
    dev_Q_Rz[  n_cells + id] += 0.5*dtodx*(dev_F_x[  n_cells + kpoimo] - dev_F_x[  n_cells + kpo])
                              + 0.5*dtody*(dev_F_y[  n_cells + kpojmo] - dev_F_y[  n_cells + kpo]);
    dev_Q_Rz[2*n_cells + id] += 0.5*dtodx*(dev_F_x[2*n_cells + kpoimo] - dev_F_x[2*n_cells + kpo])
                              + 0.5*dtody*(dev_F_y[2*n_cells + kpojmo] - dev_F_y[2*n_cells + kpo]);
    dev_Q_Rz[3*n_cells + id] += 0.5*dtodx*(dev_F_x[3*n_cells + kpoimo] - dev_F_x[3*n_cells + kpo])
                              + 0.5*dtody*(dev_F_y[3*n_cells + kpojmo] - dev_F_y[3*n_cells + kpo]);
    dev_Q_Rz[4*n_cells + id] += 0.5*dtodx*(dev_F_x[4*n_cells + kpoimo] - dev_F_x[4*n_cells + kpo])
                              + 0.5*dtody*(dev_F_y[4*n_cells + kpojmo] - dev_F_y[4*n_cells + kpo]);
    #ifdef SCALAR
    for (int i=0; i<NSCALARS; i++) {
      dev_Q_Rz[(5+i)*n_cells + id] += 0.5*dtodx*(dev_F_x[(5+i)*n_cells + kpoimo] - dev_F_x[(5+i)*n_cells + kpo])
                                + 0.5*dtody*(dev_F_y[(5+i)*n_cells + kpojmo] - dev_F_y[(5+i)*n_cells + kpo]);
    }
    #endif
    #ifdef DE
    dev_Q_Rz[(n_fields-1)*n_cells + id] += 0.5*dtodx*(dev_F_x[(n_fields-1)*n_cells + kpoimo] - dev_F_x[(n_fields-1)*n_cells + kpo])
                              + 0.5*dtody*(dev_F_y[(n_fields-1)*n_cells + kpojmo] - dev_F_y[(n_fields-1)*n_cells + kpo]);
    #endif
  }

}



#endif //CUDA
