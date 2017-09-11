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
#include"plmp_ctu_cuda.h"
#include"plmc_ctu_cuda.h"
#include"ppmp_ctu_cuda.h"
#include"ppmc_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"h_correction_2D_cuda.h"
#include"cooling_cuda.h"
#include"subgrid_routines_2D.h"



__global__ void Evolve_Interface_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F1_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F1_y,
                                           int nx, int ny, int n_ghost, Real dx, Real dy, Real dt);


Real CTU_Algorithm_2D_CUDA(Real *host_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt)
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
  int x_off_s, y_off_s; // x and y offsets for subgrid block

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
  block_tot = block1_tot*block2_tot;
  //printf("%d %d %d %d %d %d %d %d\n", nx, ny, nx_s, ny_s, block1_tot, block2_tot, remainder1, remainder2);

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
  Real *dev_conserved;
  // input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *F_x, *F_y;
  // arrays to hold the eta values for the H correction
  Real *eta_x, *eta_y, *etah_x, *etah_y;
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;


  // allocate memory on the GPU
  CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
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
  

  while (block < block_tot) {

    // calculate the global x and y offsets of this subgrid block
    // (only needed for gravitational potential)
    get_offsets_2D(nx_s, ny_s, n_ghost, x_off, y_off, block, block1_tot, block2_tot, remainder1, remainder2, &x_off_s, &y_off_s);    


    // zero all the GPU arrays
    cudaMemset(dev_conserved, 0, n_fields*BLOCK_VOL*sizeof(Real));
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
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );


    // Step 1: Do the reconstruction
    #ifdef PCM
    PCM_Reconstruction_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, nx, ny, n_ghost, gama);
    #endif
    #ifdef PLMP
    PLMP_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
    PLMP_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
    #endif
    #ifdef PLMC
    PLMC_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
    PLMC_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
    #endif
    #ifdef PPMP
    PPMP_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
    PPMP_CTU<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
    #endif
    #ifdef PPMC
    PPMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
    PPMC_cuda<<<dim2dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
    #endif
    CudaCheckError();

    #ifdef H_CORRECTION
    #ifndef CTU
    // Step 3.5: Calculate eta values for H correction
    calc_eta_x_2D<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, eta_x, nx_s, ny_s, n_ghost, gama);
    calc_eta_y_2D<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, eta_y, nx_s, ny_s, n_ghost, gama);
    CudaCheckError();
    // and etah values for each interface
    calc_etah_x_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_x, nx_s, ny_s, n_ghost);
    calc_etah_y_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_y, nx_s, ny_s, n_ghost);
    CudaCheckError();
    #endif // NO CTU
    #endif // H_CORRECTION


    // Step 2: Calculate the fluxes
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    #endif
    #ifdef HLLC 
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    #endif
    CudaCheckError();

#ifdef CTU

    // Step 3: Evolve the interface states
    Evolve_Interface_States_2D<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, nx_s, ny_s, n_ghost, dx, dy, dt);
    CudaCheckError();


    #ifdef H_CORRECTION
    // Step 3.5: Calculate eta values for H correction
    calc_eta_x_2D<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, eta_x, nx_s, ny_s, n_ghost, gama);
    calc_eta_y_2D<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, eta_y, nx_s, ny_s, n_ghost, gama);
    CudaCheckError();
    // and etah values for each interface
    calc_etah_x_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_x, nx_s, ny_s, n_ghost);
    calc_etah_y_2D<<<dim2dGrid,dim1dBlock>>>(eta_x, eta_y, etah_y, nx_s, ny_s, n_ghost);
    CudaCheckError();
    #endif


    // Step 4: Calculate the fluxes again
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
    Calculate_Exact_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
    #endif
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    Calculate_Roe_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    #endif
    #ifdef HLLC
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
    Calculate_HLLC_Fluxes_CUDA<<<dim2dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
    #endif
    CudaCheckError();

#endif //CTU


    // Step 5: Update the conserved variable array
    Update_Conserved_Variables_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, F_x, F_y, nx_s, ny_s, x_off_s, y_off_s, n_ghost, dx, dy, xbound, ybound, dt, gama);
    CudaCheckError();

    // Synchronize the total and internal energy
    #ifdef DE
    Sync_Energies_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, n_ghost, gama);
    CudaCheckError();    
    #endif

    // Apply cooling
    #ifdef COOLING_GPU
    printf("Need to fix cooling.\n");
    //cooling_kernel<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dt, gama);
    //CudaCheckError();    
    #endif

    // Step 6: Calculate the next timestep
    Calc_dt_2D<<<dim2dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, n_ghost, dx, dy, dev_dti_array, gama);
    CudaCheckError();    


    // copy the conserved variable array back to the CPU
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );

    // copy the next conserved variable blocks into appropriate buffers
    host_copy_next_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, &tmp1, n_fields);

    // copy the updated conserved variable array back into the host_conserved array on the CPU
    host_return_values_2D(nx, ny, nx_s, ny_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, n_fields);


    // copy the dti array onto the CPU
    CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, 2*ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // iterate through to find the maximum inverse dt for this subgrid block
    for (int i=0; i<2*ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }


    // add one to the counter
    block++;

  }


  // free the CPU memory
  free(host_dti_array);
  free_buffers_2D(nx, ny, nx_s, ny_s, block1_tot, block2_tot, buffer);

  // free the GPU memory
  cudaFree(dev_conserved);
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


  // return the maximum inverse timestep
  return max_dti;

}





__global__ void Evolve_Interface_States_2D(Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x, 
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           int nx, int ny, int n_ghost, Real dx, Real dy, Real dt)
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
    // right
    dev_Q_Rx[            id] += 0.5*dtody*(dev_F_y[            ipojmo] - dev_F_y[            ipo]);
    dev_Q_Rx[  n_cells + id] += 0.5*dtody*(dev_F_y[  n_cells + ipojmo] - dev_F_y[  n_cells + ipo]);
    dev_Q_Rx[2*n_cells + id] += 0.5*dtody*(dev_F_y[2*n_cells + ipojmo] - dev_F_y[2*n_cells + ipo]);
    dev_Q_Rx[3*n_cells + id] += 0.5*dtody*(dev_F_y[3*n_cells + ipojmo] - dev_F_y[3*n_cells + ipo]);
    dev_Q_Rx[4*n_cells + id] += 0.5*dtody*(dev_F_y[4*n_cells + ipojmo] - dev_F_y[4*n_cells + ipo]);
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
    // right
    dev_Q_Ry[            id] += 0.5*dtodx*(dev_F_x[            jpoimo] - dev_F_x[            jpo]); 
    dev_Q_Ry[  n_cells + id] += 0.5*dtodx*(dev_F_x[  n_cells + jpoimo] - dev_F_x[  n_cells + jpo]); 
    dev_Q_Ry[2*n_cells + id] += 0.5*dtodx*(dev_F_x[2*n_cells + jpoimo] - dev_F_x[2*n_cells + jpo]); 
    dev_Q_Ry[3*n_cells + id] += 0.5*dtodx*(dev_F_x[3*n_cells + jpoimo] - dev_F_x[3*n_cells + jpo]); 
    dev_Q_Ry[4*n_cells + id] += 0.5*dtodx*(dev_F_x[4*n_cells + jpoimo] - dev_F_x[4*n_cells + jpo]); 
  }

}


#endif //CUDA

