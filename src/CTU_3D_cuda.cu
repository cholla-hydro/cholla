#include "hip/hip_runtime.h"
/*! \file CTU_3D_cuda.cu
 *  \brief Definitions of the cuda 3D CTU algorithm functions. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<hip/hip_runtime.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"CTU_3D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_cuda.h"
#include"plmc_cuda.h"
#include"ppmp_cuda.h"
#include"ppmc_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"h_correction_3D_cuda.h"
#include"cooling_cuda.h"
#include"subgrid_routines_3D.h"
#include"io.h"


__global__ void Evolve_Interface_States_3D(Real *dev_conserved, Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           Real *dev_Q_Lz, Real *dev_Q_Rz, Real *dev_F_z,
                                           int nx, int ny, int nz, int n_ghost, 
                                           Real dx, Real dy, Real dz, Real dt, int n_fields);


Real CTU_Algorithm_3D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, int n_fields)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  //host_conserved0 contains the values at time n,
  //host_conserved1 contains the values at time n+1
  
  // Initialize dt values 
  Real max_dti = 0;
  #ifdef COOLING_GPU
  Real min_dt = 1e10;
  #endif  

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
  }

  if ( !memory_allocated ) {

    // allocate buffer to copy conserved variable blocks to/from
    if (block_tot > 1) {
      if ( NULL == ( buffer = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real)) ) ) {
        printf("Failed to allocate CPU buffer.\n");
      }
      tmp1 = buffer;
      tmp2 = buffer;
    }

    // allocate an array on the CPU to hold max_dti returned from each thread block
    host_dti_array = (Real *) malloc(ngrid*sizeof(Real));
    #ifdef COOLING_GPU
    host_dt_array = (Real *) malloc(ngrid*sizeof(Real));
    #endif

    // allocate memory on the GPU
    CudaSafeCall( hipMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Lx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Rx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Ly,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Ry,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Lz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Rz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&F_x,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&F_y,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&F_z,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( hipMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif
        
    #ifndef DYNAMIC_GPU_ALLOC 
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif 
  }  

  // counter for which block we're on
  int block = 0;


  // START LOOP OVER SUBGRID BLOCKS
  while (block < block_tot) {

    // copy the conserved variable block to the buffer
    host_copy_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved0, buffer, n_fields);

    get_offsets_3D(nx_s, ny_s, nz_s, n_ghost, x_off, y_off, z_off, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, &x_off_s, &y_off_s, &z_off_s);

    // copy the conserved variables onto the GPU
    CudaSafeCall( hipMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), hipMemcpyHostToDevice) );
      

    // Step 1: Do the reconstruction
    #ifdef PCM
    hipLaunchKernelGGL(PCM_Reconstruction_3D, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    #endif //PCM
    #ifdef PLMP
    hipLaunchKernelGGL(PLMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PLMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PLMP 
    #ifdef PLMC
    hipLaunchKernelGGL(PLMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PLMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PLMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PLMC 
    #ifdef PPMP
    hipLaunchKernelGGL(PPMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PPMP_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMP
    #ifdef PPMC
    hipLaunchKernelGGL(PPMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    hipLaunchKernelGGL(PPMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    hipLaunchKernelGGL(PPMC_cuda, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMC
    CudaCheckError();
   

    // Step 2: Calculate the fluxes
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //ROE
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    CudaCheckError();


    #ifdef CTU
    // Step 3: Evolve the interface states
    hipLaunchKernelGGL(Evolve_Interface_States_3D, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dt, n_fields);
    CudaCheckError();


    // Step 4: Calculate the fluxes again
    #ifdef EXACT
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //ROE
    #ifdef HLLC
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    CudaCheckError();
    #endif //CTU

  
    // Step 5: Update the conserved variable array
    hipLaunchKernelGGL(Update_Conserved_Variables_3D, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, F_x, F_y, F_z, nx_s, ny_s, nz_s, x_off, y_off, z_off, n_ghost, dx, dy, dz, xbound, ybound, zbound, dt, gama, n_fields);
    CudaCheckError();


    // Synchronize the total and internal energies
    #ifdef DE
    hipLaunchKernelGGL(Sync_Energies_3D, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #endif


    // Apply cooling
    #ifdef COOLING_GPU
    hipLaunchKernelGGL(cooling_kernel, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, dt, gama, dev_dt_array);
    CudaCheckError();
    #endif


    // Step 6: Calculate the next timestep
    hipLaunchKernelGGL(Calc_dt_3D, dim3(dim1dGrid), dim3(dim1dBlock), 0, 0, dev_conserved, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dev_dti_array, gama);
    CudaCheckError();

  

    // copy the updated conserved variable array back to the CPU
    CudaSafeCall( hipMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), hipMemcpyDeviceToHost) );
    CudaCheckError();

    // copy the updated conserved variable array from the buffer into the host_conserved array on the CPU
    host_return_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved1, buffer, n_fields);

    // copy the dti array onto the CPU
    CudaSafeCall( hipMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), hipMemcpyDeviceToHost) );
    // iterate through to find the maximum inverse dt for this subgrid block
    for (int i=0; i<ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }
    #ifdef COOLING_GPU
    // copy the dt array from cooling onto the CPU
    CudaSafeCall( hipMemcpy(host_dt_array, dev_dt_array, ngrid*sizeof(Real), hipMemcpyDeviceToHost) );
    // find maximum inverse timestep from cooling time
    for (int i=0; i<ngrid; i++) {
      min_dt = fmin(min_dt, host_dt_array[i]);
    }  
    if (min_dt < C_cfl/max_dti) {
      max_dti = C_cfl/min_dt;
    }
    #endif

    // add one to the counter
    block++;

  }


  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_CTU_3D();
  #endif


  // return the maximum inverse timestep
  return max_dti;

}


void Free_Memory_CTU_3D() {

  // free CPU memory
  if (block_tot > 1) free(buffer);
  free(host_dti_array);
  #ifdef COOLING_GPU
  free(host_dt_array);  
  #endif

  // free the GPU memory
  hipFree(dev_conserved);
  hipFree(Q_Lx);
  hipFree(Q_Rx);
  hipFree(Q_Ly);
  hipFree(Q_Ry);
  hipFree(Q_Lz);
  hipFree(Q_Rz);
  hipFree(F_x);
  hipFree(F_y);
  hipFree(F_z);
  hipFree(dev_dti_array);
  #ifdef COOLING_GPU
  hipFree(dev_dt_array);
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
