#include "hip/hip_runtime.h"
/*! \file CTU_1D_cuda.cu
 *  \brief Definitions of the cuda CTU algorithm functions. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<hip/hip_runtime.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"CTU_1D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_cuda.h"
#include"plmc_cuda.h"
#include"ppmp_cuda.h"
#include"ppmc_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"cooling_cuda.h"
#include"error_handling.h"
#include"io.h"



Real CTU_Algorithm_1D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt, int n_fields)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //host_conserved0 contains the values at time n
  //host_conserved1 will contain the values at time n+1

  // Initialize dt values
  Real max_dti = 0;
  #ifdef COOLING_GPU
  Real min_dt = 1e10;
  #endif  

  int n_cells = nx;
  int ny = 1;
  int nz = 1;

  // set the dimensions of the cuda grid
  ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dimGrid(ngrid, 1, 1);
  dim3 dimBlock(TPB, 1, 1);

  if ( !memory_allocated ) {

    // allocate an array on the CPU to hold max_dti returned from each thread block
    host_dti_array = (Real *) malloc(ngrid*sizeof(Real));
    #ifdef COOLING_GPU
    host_dt_array = (Real *) malloc(ngrid*sizeof(Real));
    #endif

    // allocate memory on the GPU
    CudaSafeCall( hipMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Lx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&Q_Rx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&F_x,   (n_fields)*n_cells*sizeof(Real)) );
    CudaSafeCall( hipMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #if defined COOLING_GPU
    CudaSafeCall( hipMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif  

    #ifndef DYNAMIC_GPU_ALLOC 
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif 
  }

  // copy the conserved variable array onto the GPU
  CudaSafeCall( hipMemcpy(dev_conserved, host_conserved0, n_fields*n_cells*sizeof(Real), hipMemcpyHostToDevice) );
  CudaCheckError();


  // Step 1: Do the reconstruction
  #ifdef PCM
  hipLaunchKernelGGL(PCM_Reconstruction_1D, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif
  #ifdef PLMP
  hipLaunchKernelGGL(PLMP_cuda, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PLMC
  hipLaunchKernelGGL(PLMC_cuda, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PPMP
  hipLaunchKernelGGL(PPMP_cuda, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PPMC
  hipLaunchKernelGGL(PPMC_cuda, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif

  
  // Step 2: Calculate the fluxes
  #ifdef EXACT
  hipLaunchKernelGGL(Calculate_Exact_Fluxes_CUDA, dim3(dimGrid), dim3(dimBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef ROE
  hipLaunchKernelGGL(Calculate_Roe_Fluxes_CUDA, dim3(dimGrid), dim3(dimBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef HLLC 
  hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dim3(dimGrid), dim3(dimBlock), 0, 0, Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  CudaCheckError();


  // Step 3: Update the conserved variable array
  hipLaunchKernelGGL(Update_Conserved_Variables_1D, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, F_x, n_cells, x_off, n_ghost, dx, xbound, dt, gama, n_fields);
  CudaCheckError();
   

  // Sychronize the total and internal energy, if using dual-energy formalism
  #ifdef DE
  hipLaunchKernelGGL(Sync_Energies_1D, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, n_cells, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif


  // Apply cooling
  #ifdef COOLING_GPU
  hipLaunchKernelGGL(cooling_kernel, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gama, dev_dti_array);
  CudaCheckError();
  #endif

  // Calculate the next timestep
  hipLaunchKernelGGL(Calc_dt_1D, dim3(dimGrid), dim3(dimBlock), 0, 0, dev_conserved, n_cells, n_ghost, dx, dev_dti_array, gama);
  CudaCheckError();


  // copy the conserved variable array back to the CPU
  CudaSafeCall( hipMemcpy(host_conserved1, dev_conserved, n_fields*n_cells*sizeof(Real), hipMemcpyDeviceToHost) );

  // copy the dti array onto the CPU
  CudaSafeCall( hipMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), hipMemcpyDeviceToHost) );
  // iterate through to find the maximum inverse dt for this subgrid block
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, host_dti_array[i]);
  }
  #if defined COOLING_GPU
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

  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_CTU_1D();
  #endif
  

  // return the maximum inverse timestep
  return max_dti;


}

void Free_Memory_CTU_1D() {

  // free the CPU memory
  free(host_dti_array);
  #if defined COOLING_GPU
  free(host_dt_array);  
  #endif

  // free the GPU memory
  hipFree(dev_conserved);
  hipFree(Q_Lx);
  hipFree(Q_Rx);
  hipFree(F_x);
  hipFree(dev_dti_array);
  #if defined COOLING_GPU
  hipFree(dev_dt_array);
  #endif

}


#endif //CUDA
