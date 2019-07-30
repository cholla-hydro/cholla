/*! \file CTU_2D_cuda.cu
 *  \brief Definitions of the cuda 2D CTU algorithm functions. */

#ifdef CUDA

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"CTU_2D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_cuda.h"
#include"plmc_cuda.h"
#include"ppmp_cuda.h"
#include"ppmc_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"h_correction_2D_cuda.h"
#include"cooling_cuda.h"
#include"subgrid_routines_2D.h"



__global__ void Evolve_Interface_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F1_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F1_y,
                                           int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, int n_fields);


Real CTU_Algorithm_2D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, int n_fields)
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
    // calculate the dimensions for each subgrid block
    sub_dimensions_2D(nx, ny, n_ghost, &nx_s, &ny_s, &block1_tot, &block2_tot, &remainder1, &remainder2, n_fields);
    //printf("%d %d %d %d %d %d\n", nx_s, ny_s, block1_tot, block2_tot, remainder1, remainder2);
    nz_s = 1;
    block_tot = block1_tot*block2_tot;
    // number of cells in one subgrid block
    BLOCK_VOL = nx_s*ny_s*nz_s;
    // dimensions for the 1D GPU grid
    ngrid = (BLOCK_VOL + TPB - 1) / (TPB);
    #ifndef DYNAMIC_GPU_ALLOC
    block_size = true;
    #endif
  }
  // set values for GPU kernels
  // number of blocks per 1D grid  
  dim3 dim2dGrid(ngrid, 1, 1);
  //number of threads per 1D block   
  dim3 dim1dBlock(TPB, 1, 1);

  // Set up pointers for the location to copy from and to
  if (block_tot == 1) {
    tmp1 = host_conserved0;
    tmp2 = host_conserved1;
  }  

  if ( !memory_allocated ) {

    // allocate buffer to copy conserved variable blocks from and to 
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
    CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ly, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&Q_Ry, n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_x,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&F_y,  n_fields*BLOCK_VOL*sizeof(Real)) );
    CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
    #ifdef COOLING_GPU
    CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
    #endif 

    #ifndef DYNAMIC_GPU_ALLOC 
    // If memory is single allocated: memory_allocated becomes true and succesive timesteps won't allocate memory.
    // If the memory is not single allocated: memory_allocated remains Null and memory is allocated every timestep.
    memory_allocated = true;
    #endif 
  }

  // counter for which block we're on
  int block = 0;


  // START LOOP OVER SUBGRID BLOCKS HERE
  while (block < block_tot) {

    // copy the conserved variable block to the buffer
    host_copy_block_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved0, buffer, n_fields);

    // calculate the global x and y offsets of this subgrid block
    // (only needed for gravitational potential)
    get_offsets_2D(nx_s, ny_s, n_ghost, x_off, y_off, block, block1_tot, block2_tot, remainder1, remainder2, &x_off_s, &y_off_s);    

    // copy the conserved variables onto the GPU
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );


    // Step 1: Do the reconstruction
    #ifdef PCM
    PCM_Reconstruction_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx_s, ny_s, n_ghost, gama, n_fields);
    #endif
    #ifdef PLMP
    PLMP_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMP_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    #ifdef PLMC
    PLMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    #ifdef PPMP
    PPMP_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PPMP_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    #ifdef PPMC
    PPMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PPMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    #endif
    CudaCheckError();


    // Step 2: Calculate the fluxes
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef HLLC 
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    CudaCheckError();

#ifdef CTU

    // Step 3: Evolve the interface states
    Evolve_Interface_States_2D<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, nx_s, ny_s, n_ghost, dx, dy, dt, n_fields);
    CudaCheckError();


    // Step 4: Calculate the fluxes again
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    #ifdef HLLC
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #endif
    CudaCheckError();

#endif //CTU


    // Step 5: Update the conserved variable array
    Update_Conserved_Variables_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, F_x, F_y, nx_s, ny_s, x_off_s, y_off_s, n_ghost, dx, dy, xbound, ybound, dt, gama, n_fields);
    CudaCheckError();

    // Synchronize the total and internal energy
    #ifdef DE
    Sync_Energies_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, n_ghost, gama, n_fields);
    CudaCheckError();    
    #endif

    // Apply cooling
    #ifdef COOLING_GPU
    cooling_kernel<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, dt, gama, dev_dt_array);
    CudaCheckError();    
    #endif

    // Step 6: Calculate the next timestep
    Calc_dt_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, n_ghost, dx, dy, dev_dti_array, gama);
    CudaCheckError();    


    // copy the conserved variable array back to the CPU
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );

    // copy the updated conserved variable array back into the host_conserved array on the CPU
    host_return_block_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved1, buffer, n_fields);


    // copy the dti array onto the CPU
    CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // iterate through to find the maximum inverse dt for this subgrid block
    for (int i=0; i<ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }
    #ifdef COOLING_GPU
    // copy the dt array from cooling onto the CPU
    CudaSafeCall( cudaMemcpy(host_dt_array, dev_dt_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // iterate through to find the minimum dt for this subgrid block
    for (int i=0; i<ngrid; i++) {
      min_dt = fmin(min_dt, host_dt_array[i]);
    }  
    //printf("%f %f\n", min_dt, 0.3/max_dti); 
    if (min_dt < 0.3/max_dti) {
      //printf("%f %f\n", min_dt, 0.3/max_dti); 
      min_dt = fmax(min_dt, 1.0);
      max_dti = 0.3/min_dt;
    }
    #endif


    // add one to the counter
    block++;

  }


  #ifdef DYNAMIC_GPU_ALLOC
  // If memory is not single allocated then free the memory every timestep.
  Free_Memory_CTU_2D();
  #endif

  // return the maximum inverse timestep
  return max_dti;

}

void Free_Memory_CTU_2D() {

  // free the CPU memory
  if (block_tot > 1) free(buffer);
  free(host_dti_array);
  #ifdef COOLING_GPU
  free(host_dt_array);  
  #endif    

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif

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

