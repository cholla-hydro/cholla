/*! \file VL_2D_cuda.cu
 *  \brief Definitions of the cuda 2D VL algorithm functions. */

#ifdef CUDA
#ifdef VL

#include<stdio.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"VL_2D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_vl_cuda.h"
#include"ppmp_vl_cuda.h"
#include"ppmc_vl_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"h_correction_2D_cuda.h"
#include"cooling.h"
#include"subgrid_routines_2D.h"

//#define TIME
//#define TEST


__global__ void Update_Conserved_Variables_2D_notime(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                              int n_ghost, Real dx, Real dy, Real dt, Real gamma);

__global__ void Update_Conserved_Variables_2D_wtime(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                              int n_ghost, Real dx, Real dy, Real dt, Real *dti_array, Real gamma);


Real VL_Algorithm_2D_CUDA(Real *host_conserved, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt)
{

  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array

  #ifdef TIME
  // capture the start time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  #endif

  int n_fields = 5;
  #ifdef DE
  n_fields++;
  #endif

  // dimensions of subgrid blocks
  int nx_s; //number of cells in the subgrid block along x direction
  int ny_s; //number of cells in the subgrid block along y direction
  int nz_s = 1; //number of cells in the subgrid block along z direction

  // total number of blocks needed
  int block_tot;    //total number of subgrid blocks (unsplit == 1)
  int block1_tot;   //total number of subgrid blocks in x direction
  int block2_tot;   //total number of subgrid blocks in y direction
  int remainder1;   //modulus of number of cells after block subdivision in x direction
  int remainder2;   //modulus of number of cells after block subdivision in y direction 

  // counter for which block we're on
  int block = 0;

  // calculate the dimensions for each subgrid block
  sub_dimensions_2D(nx, ny, n_ghost, &nx_s, &ny_s, &block1_tot, &block2_tot, &remainder1, &remainder2, n_fields);
  //printf("%d %d %d %d %d %d\n", nx_s, ny_s, block1_tot, block2_tot, remainder1, remainder2);
  block_tot = block1_tot*block2_tot;

  // number of cells in one subgrid block
  int BLOCK_VOL = nx_s*ny_s*nz_s;

  // define the dimensions for the 2D grid
  //int  ngrid = (n_cells + TPB - 1) / TPB;
  int  ngrid = (BLOCK_VOL + 2*TPB - 1) / (2*TPB);

  //number of blocks per 2-d grid  
  dim3 dim2dGrid(ngrid, 2, 1);

  //number of threads per 1-d block   
  dim3 dim1dBlock(TPB, 1, 1);

  // allocate buffer arrays to copy conserved variable slices into
  Real **buffer;
  allocate_buffers_2D(block1_tot, block2_tot, BLOCK_VOL, buffer, n_fields);
  // and set up pointers for the location to copy from and to
  Real *tmp1;
  Real *tmp2;

  // allocate an array on the CPU to hold max_dti returned from each thread block
  Real max_dti = 0;
  Real *host_dti_array;
  host_dti_array = (Real *) malloc(2*ngrid*sizeof(Real));

  // allocate GPU arrays
  // conserved variables
  Real *dev_conserved, *dev_conserved_half;
  // input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *F_x, *F_y;
  // arrays to hold the eta values for the H correction
  Real *eta_x, *eta_y, *etah_x, *etah_y;
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;


#ifdef TEST
  Real *test1, *test2;
  test1 = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real));
  test2 = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real));
#endif


  // allocate memory on the GPU
  CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Lx, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Rx, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ly, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ry, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_x,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_y,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_x,   BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_y,   BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_x,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_y,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_dti_array, 2*ngrid*sizeof(Real)) );
  

  // transfer first conserved variable slice into the first buffer
  host_copy_init_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, remainder1, BLOCK_VOL, host_conserved, buffer, &tmp1, &tmp2, n_fields);
  
  // START LOOP OVER SUBGRID BLOCKS HERE
  while (block < block_tot) {

    // zero all the GPU arrays
    cudaMemset(dev_conserved, 0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(dev_conserved_half, 0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(Q_Lx,  0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(Q_Rx,  0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(Q_Ly,  0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(Q_Ry,  0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(F_x,   0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(F_y,   0, n_fields*BLOCK_VOL*sizeof(Real));
    cudaMemset(eta_x,  0,  BLOCK_VOL*sizeof(Real));
    cudaMemset(eta_y,  0,  BLOCK_VOL*sizeof(Real));
    cudaMemset(etah_x, 0,  BLOCK_VOL*sizeof(Real));
    cudaMemset(etah_y, 0,  BLOCK_VOL*sizeof(Real));
    cudaMemset(dev_dti_array, 0, 2*ngrid*sizeof(Real));
    CudaCheckError();

    // copy the conserved variables onto the GPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU copy: %5.3f ms\n", elapsedTime);
    #endif


    // Step 1: Use PCM reconstruction to put conserved variables into interface arrays
    PCM_Reconstruction_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx_s, ny_s, n_ghost, gama);
    CudaCheckError();

    // Step 2: Calculate first-order upwind fluxes 
    #ifdef EXACT
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    Calculate_Exact_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    #ifdef TIME
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("x fluxes:  %5.3f ms\n", elapsedTime);
    #endif
    CudaCheckError();  
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    Calculate_Exact_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    #ifdef TIME
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("y fluxes:  %5.3f ms\n", elapsedTime);
    #endif  
    CudaCheckError();  
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    CudaCheckError();
    Calculate_Roe_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    CudaCheckError();
    #endif



    // Step 3: Update the conserved variables half a timestep 
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif    
    Update_Conserved_Variables_2D_notime<<<dim2dGrid,dim1dBlock>>>(dev_conserved, dev_conserved_half, F_x, F_y, nx_s, ny_s, n_ghost, dx, dy, 0.5*dt, gama);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("conserved variable update: %5.3f ms\n", elapsedTime);
    #endif     


    // Step 4: Construct left and right interface values using updated conserved variables
    #ifdef PLMP
    PLMP_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    CudaCheckError();
    PLMP_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    CudaCheckError();
    #endif
    #ifdef PLMC
    printf("PLMC not supported for Van Leer integrator.\n");
    #endif
    #ifdef PPMP
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    PPMP_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to do x reconstruction: %5.3f ms\n", elapsedTime);
    cudaEventRecord(start, 0);
    #endif
    PPMP_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to do y reconstruction: %5.3f ms\n", elapsedTime);
    #endif
    #endif //PPMP
    #ifdef PPMC
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    PPMC_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to do x reconstruction: %5.3f ms\n", elapsedTime);
    cudaEventRecord(start, 0);
    #endif
    PPMC_VL<<<dim2dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time to do y reconstruction: %5.3f ms\n", elapsedTime);
    #endif
    #endif //PPMC
#ifdef TEST 
    CudaSafeCall( cudaMemcpy(test1, Q_Lx, 5*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
    CudaSafeCall( cudaMemcpy(test2, Q_Ly, 5*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
    for (int i=0; i<nx; i++) {
      for (int j=0; j<ny; j++) {
        if (test1[i + j*nx + 1*nx*ny] != test2[j + i*nx + 2*nx*ny]) {
          printf("%3d %3d %f %f\n", i, j, test1[i + j*nx + 1*nx*ny], test2[j + i*nx + 2*nx*ny]);
        }
      }
    }
#endif

    #ifdef H_CORRECTION
    // Step 4.5: Calculate eta values for H correction
    calc_eta_x_2D<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, eta_x, nx_s, ny_s, n_ghost, gama);
    calc_eta_y_2D<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, eta_y, nx_s, ny_s, n_ghost, gama);
    CudaCheckError();
    // and etah values for each interface
    calc_etah_x_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_x, nx_s, ny_s, n_ghost);
    calc_etah_y_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_y, nx_s, ny_s, n_ghost);
    CudaCheckError();
    #endif


    // Step 5: Calculate the fluxes again
    #ifdef EXACT
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    Calculate_Exact_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    #ifdef TIME
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("x fluxes:  %5.3f ms\n", elapsedTime);
    #endif  
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    Calculate_Exact_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    #ifdef TIME
    // get stop time, and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("y fluxes:  %5.3f ms\n", elapsedTime);
    #endif    
    CudaCheckError();
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    Calculate_Roe_Fluxes<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    CudaCheckError();
    #endif


    // Step 5: Update the conserved variable array
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif    
    Update_Conserved_Variables_2D_wtime<<<dim2dGrid,dim1dBlock>>>(dev_conserved, F_x, F_y, nx_s, ny_s, n_ghost, dx, dy, dt, dev_dti_array, gama);
    CudaCheckError();
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("conserved variable update: %5.3f ms\n", elapsedTime);
    #endif     

    #ifdef COOLING_GPU
    cooling_kernel<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dt, gama);
    #endif


    // copy the conserved variable array back to the CPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("GPU return: %5.3f ms\n", elapsedTime);
    #endif


    // copy the next conserved variable blocks into appropriate buffers
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif    
    host_copy_next_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, &tmp1, n_fields);


    // copy the updated conserved variable array back into the host_conserved array on the CPU
    host_return_values_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, n_fields);
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CPU copying: %5.3f ms\n", elapsedTime);
    #endif     


    // copy the dti array onto the CPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif      
    CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, 2*ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // iterate through to find the maximum inverse dt for this subgrid block
    for (int i=0; i<2*ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }
    #ifdef TIME
    // get stop time and display the timing results
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("dti copying & calc: %5.3f ms\n", elapsedTime);
    #endif     


    // add one to the counter
    block++;

  }


  // free the CPU memory
  free(host_dti_array);
  free_buffers_2D(nx, ny, nx_s, ny_s, block1_tot, block2_tot, buffer);

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(eta_x);
  cudaFree(eta_y);
  cudaFree(etah_x);
  cudaFree(etah_y);
  cudaFree(dev_dti_array);

  #ifdef TIME
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  #endif

 
#ifdef TEST
  free(test1);
  free(test2);
#endif

  // return the maximum inverse timestep
  return max_dti;

}


__global__ void Update_Conserved_Variables_2D_notime(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real gamma)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;

  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;


  // all threads but one outer ring of ghost cells 
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1)
  {
    // update the conserved variable array
    imo = xid-1 + yid*nx;
    jmo = xid + (yid-1)*nx;
    dev_conserved_half[            id] = dev_conserved[            id] + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]) 
                                                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]) 
                                                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]); 
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
  } 
}



__global__ void Update_Conserved_Variables_2D_wtime(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny, int n_ghost, Real dx, Real dy, Real dt, Real *dti_array, Real gamma)
{
  __shared__ Real max_dti[TPB];

  Real d, d_inv, vx, vy, vz, P, cs;
  int id, tid, xid, yid, n_cells;
  int imo, jmo;

  Real dtodx = dt/dx;
  Real dtody = dt/dy;

  n_cells = nx*ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y*gridDim.x;
  id = threadIdx.x + blockId * blockDim.x;
  yid = id / nx;
  xid = id - yid*nx;
  // and a thread id within the block
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost)
  {
    // update the conserved variable array
    imo = xid-1 + yid*nx;
    jmo = xid + (yid-1)*nx;
    dev_conserved[            id] += dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                  +  dtody * (dev_F_y[            jmo] - dev_F_y[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id]) 
                                  +  dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id]) 
                                  +  dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id]); 
    dev_conserved[3*n_cells + id] += dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                  +  dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                  +  dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id]);
   

    // start timestep calculation here
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    P  = fmax(P, (Real) 1.0e-20);
    // find the max wavespeed in that cell, use it to calculate the inverse timestep
    cs = sqrt(d_inv * gamma * P);
    max_dti[tid] = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
  }
  __syncthreads();
  
  // do the reduction in shared memory (find the max inverse timestep in the block)
  for (unsigned int s=1; s<blockDim.x; s*=2) {
    if (tid % (2*s) == 0) {
      max_dti[tid] = fmax(max_dti[tid], max_dti[tid + s]);
    }
    __syncthreads();
  }

  // write the result for this block to global memory
  if (tid == 0) dti_array[blockId] = max_dti[0];
  
}

#endif //VL
#endif //CUDA

