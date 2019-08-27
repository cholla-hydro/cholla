/*! \file VL_3D_cuda.cu
 *  \brief Definitions of the cuda 3D VL algorithm functions. */

#ifdef CUDA
#ifdef SIMPLE

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"simple_3D_cuda.h"
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
#include"hll_cuda.h"



Real Simple_Algorithm_3D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, int n_fields, Real density_floor, Real U_floor,  Real *host_grav_potential)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  //host_conserved0 contains the values at time n,
  //host_conserved1 will contain the values at time n+1

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
    //host_grav_potential is NULL if not using GRAVITY
    temp_potential = host_grav_potential;
  }

  if ( !memory_allocated ){

    // allocate buffer to copy conserved variable blocks to/from
    if (block_tot > 1) {
      if ( NULL == ( buffer = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real)) ) ) {
        printf("Failed to allocate CPU buffer.\n");
      }
      tmp1 = buffer;
      tmp2 = buffer;
      
      #if defined( GRAVITY ) && defined( GRAVITY_COUPLE_GPU )
      if ( NULL == ( buffer_potential = (Real *) malloc(BLOCK_VOL*sizeof(Real)) ) ) {
        printf("Failed to allocate CPU Grav_Potential buffer.\n");
      }
      #else
      buffer_potential = NULL;
      #endif
      temp_potential = buffer_potential;
    }
    // allocate an array on the CPU to hold max_dti returned from each thread block
    host_dti_array = (Real *) malloc(ngrid*sizeof(Real));
    #ifdef COOLING_GPU
    host_dt_array = (Real *) malloc(ngrid*sizeof(Real));
    #endif  

    // allocate memory on the GPU
    CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
    // CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rz,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_z,   n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif 
    
    #if defined( GRAVITY ) && defined( GRAVITY_COUPLE_GPU )
    CudaSafeCall( cudaMalloc((void**)&dev_grav_potential, BLOCK_VOL*sizeof(Real)) );
    #else
    dev_grav_potential = NULL;
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
    host_copy_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved0, buffer, n_fields, host_grav_potential, buffer_potential);

    // calculate the global x, y, and z offsets of this subgrid block
    get_offsets_3D(nx_s, ny_s, nz_s, n_ghost, x_off, y_off, z_off, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, &x_off_s, &y_off_s, &z_off_s);

    // copy the conserved variables onto the GPU
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #if defined( GRAVITY ) && defined( GRAVITY_COUPLE_GPU )
    CudaSafeCall( cudaMemcpy(dev_grav_potential, temp_potential, BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #endif
 
  
    // Step 1: Construct left and right interface values using updated conserved variables
    #ifdef PCM
    PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    #endif
    #ifdef PLMP
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PLMP 
    #ifdef PLMC
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);  
    #endif
    #ifdef PPMP
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMP
    CudaCheckError();
    #ifdef PPMC
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    CudaCheckError();
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    CudaCheckError();
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    CudaCheckError();
    #endif //PPMC
    

    // Step 2: Calculate the fluxes again
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //ROE
    #ifdef HLLC 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLLC
    #ifdef HLL 
    Calculate_HLL_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_HLL_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_HLL_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //HLL
    CudaCheckError();
    
    #ifdef DE
    // Compute the divergence of Vel before updating the conserved array, this solves sincronization issues when adding this term on Update_Conserved_Variables_3D
    Add_Pressure_Div_V_to_Advected_Internal_Energy<<<dim1dGrid,dim1dBlock>>>( dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, F_x, F_y, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz,  dt, gama, n_fields );
    #endif

    // Step 3: Update the conserved variable array
    Update_Conserved_Variables_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved,  Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, F_x, F_y, F_z, nx_s, ny_s, nz_s, x_off_s, y_off_s, z_off_s, n_ghost, dx, dy, dz, xbound, ybound, zbound, dt, gama, n_fields, density_floor, dev_grav_potential);
    CudaCheckError();

    #ifndef GRAVITY_COUPLE_CPU //If gravity is added on the CPU, sync Eergies on the CPU after gravity has been added
    #ifdef DE
    Select_Internal_Energy_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    Sync_Energies_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #endif
    
    #ifdef TEMPERATURE_FLOOR
    Apply_Temperature_Floor<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, U_floor );
    CudaCheckError();
    #endif //TEMPERATURE_FLOOR
    #endif//GRAVITY_COUPLE_CPU

    // Apply cooling
    #ifdef COOLING_GPU
    cooling_kernel<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, dt, gama, dev_dt_array);  
    CudaCheckError();
    #endif
 
    // Step 4: Calculate the next time step
    #ifndef GRAVITY_COUPLE_CPU //If gravity is coupled on the CPU, dt is computed on the CPU after hydro update
    Calc_dt_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dev_dti_array, gama);
    CudaCheckError();
    #endif

    // copy the updated conserved variable array back to the CPU
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );

    // copy the updated conserved variable array from the buffer into the host_conserved array on the CPU
    host_return_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved1, buffer, n_fields);

    // copy the dti array onto the CPU
    CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // find maximum inverse timestep from CFL condition
    for (int i=0; i<ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }
    #ifdef COOLING_GPU
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

    // add one to the counter
    block++;

  }

  
  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_Simple_3D();
  #endif


  // return the maximum inverse timestep
  return max_dti;

}


void Free_Memory_Simple_3D(){
  
  // free CPU memory
  if (block_tot > 1) free(buffer);
  free(host_dti_array);  
  #ifdef COOLING_GPU
  free(host_dt_array);  
  #endif  
  
  // free the GPU memory
  cudaFree(dev_conserved);
  // cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(Q_Lz);
  cudaFree(Q_Rz);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(F_z);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif

}




#endif //SIMPLE
#endif //CUDA
