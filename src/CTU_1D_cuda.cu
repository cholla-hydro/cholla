/*! \file CTU_1D_cuda.cu
 *  \brief Definitions of the cuda CTU algorithm functions. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
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
    CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,   (n_fields)*n_cells*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #if defined COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif  

    #ifndef DYNAMIC_GPU_ALLOC 
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif 
  }

  // copy the conserved variable array onto the GPU
  CudaSafeCall( cudaMemcpy(dev_conserved, host_conserved0, n_fields*n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  CudaCheckError();


  // Step 1: Do the reconstruction
  #ifdef PCM
  PCM_Reconstruction_1D<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif
  #ifdef PLMP
  PLMP_cuda<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PLMC
  PLMC_cuda<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PPMP
  PPMP_cuda<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif
  #ifdef PPMC
  PPMC_cuda<<<dimGrid,dimBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx, ny, nz, n_ghost, dx, dt, gama, 0, n_fields);
  CudaCheckError();
  #endif

  
  // Step 2: Calculate the fluxes
  #ifdef EXACT
  Calculate_Exact_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  #ifdef HLLC 
  Calculate_HLLC_Fluxes_CUDA<<<dimGrid,dimBlock>>>(Q_Lx, Q_Rx, F_x, nx, ny, nz, n_ghost, gama, 0, n_fields);
  #endif
  CudaCheckError();

  #ifdef DE
  // Compute the divergence of Vel before updating the conserved array, this solves syncronization issues when adding this term on Update_Conserved_Variables
  Partial_Update_Advected_Internal_Energy_1D<<<dimGrid,dimBlock>>>( dev_conserved, Q_Lx, Q_Rx, nx, n_ghost, dx, dt, gama, n_fields );
  #endif


  // Step 3: Update the conserved variable array
  Update_Conserved_Variables_1D<<<dimGrid,dimBlock>>>(dev_conserved, F_x, n_cells, x_off, n_ghost, dx, xbound, dt, gama, n_fields);
  CudaCheckError();
   

  // Sychronize the total and internal energy, if using dual-energy formalism
  #ifdef DE
  Select_Internal_Energy_1D<<<dimGrid,dimBlock>>>(dev_conserved, nx, n_ghost, n_fields);  
  Sync_Energies_1D<<<dimGrid,dimBlock>>>(dev_conserved, n_cells, n_ghost, gama, n_fields);
  CudaCheckError();
  #endif


  // Apply cooling
  #ifdef COOLING_GPU
  cooling_kernel<<<dimGrid,dimBlock>>>(dev_conserved, nx, ny, nz, n_ghost, n_fields, dt, gama, dev_dti_array);
  CudaCheckError();
  #endif

  // Calculate the next timestep
  Calc_dt_1D<<<dimGrid,dimBlock>>>(dev_conserved, n_cells, n_ghost, dx, dev_dti_array, gama);
  CudaCheckError();


  // copy the conserved variable array back to the CPU
  CudaSafeCall( cudaMemcpy(host_conserved1, dev_conserved, n_fields*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );

  // copy the dti array onto the CPU
  CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // iterate through to find the maximum inverse dt for this subgrid block
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, host_dti_array[i]);
  }
  #if defined COOLING_GPU
  // copy the dt array from cooling onto the CPU
  CudaSafeCall( cudaMemcpy(host_dt_array, dev_dt_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
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
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(F_x);
  cudaFree(dev_dti_array);
  #if defined COOLING_GPU
  cudaFree(dev_dt_array);
  #endif

}


#endif //CUDA
