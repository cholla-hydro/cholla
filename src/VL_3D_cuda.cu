/*! \file VL_3D_cuda.cu
 *  \brief Definitions of the cuda 3D VL algorithm functions. */

#ifdef CUDA
#ifdef VL

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"hydro_cuda.h"
#include"VL_3D_cuda.h"
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

//#define TIME

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields);



Real VL_Algorithm_3D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost, Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt, int n_fields)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array
  //host_conserved0 contains the values at time n,
  //host_conserved1 will contain the values at time n+1

#ifdef TIME
  cudaEvent_t start_VL, stop_VL;
  cudaEvent_t start, stop;
  cudaEventCreate(&start_VL);
  cudaEventCreate(&stop_VL);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start_VL, 0);
  float elapsedTime;
  Real cpto, cpfr;
  Real buff, dti;
  Real ppmx, ppmy, ppmz;
  Real r1x, r1y, r1z;
  Real r2x, r2y, r2z;
  Real pcm, ie, cvu;
  Real cool;
  Real arrays, other;
  cpto = cpfr = 0;
  buff = dti = 0;
  ppmx = ppmy = ppmz = 0;
  r1x = r1y = r1z = 0;
  r2x = r2y = r2z = 0;
  pcm = ie = cvu = 0;
  cool = 0;
  arrays = other = 0;
#endif

  // dimensions of subgrid blocks
  int nx_s, ny_s, nz_s; 
  // x, y, and z offsets for subgrid blocks
  int x_off_s, y_off_s, z_off_s;
  // total number of subgrid blocks needed
  int block_tot;
  // number of subgrid blocks needed in each direction
  int block1_tot, block2_tot, block3_tot;
  // modulus of number of cells after block subdivision in each direction
  int remainder1, remainder2, remainder3;

  // counter for which block we're on
  int block = 0;

  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  // calculate the dimensions for the subgrid blocks
  sub_dimensions_3D(nx, ny, nz, n_ghost, &nx_s, &ny_s, &nz_s, &block1_tot, &block2_tot, &block3_tot, &remainder1, &remainder2, &remainder3, n_fields);
  //nx_s =ny_s = nz_s= 264;
  //block1_tot = block2_tot = block3_tot = 1;
  //remainder1 = remainder2 = remainder3 = 0;
  //printf("Subgrid dimensions set: %d %d %d %d %d %d %d %d %d\n", nx_s, ny_s, nz_s, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3);
  //fflush(stdout);
  block_tot = block1_tot*block2_tot*block3_tot;
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  other += elapsedTime;
  #endif //TIME

  // number of cells in one subgrid block
  int BLOCK_VOL = nx_s*ny_s*nz_s;

  // dimensions for the 1D GPU grid
  int  ngrid = (BLOCK_VOL + TPB - 1) / TPB;
  //printf("ngrid: %d\n", ngrid);
  //fflush(stdout);

  //number of blocks per 1D grid  
  dim3 dim1dGrid(ngrid, 1, 1);

  //number of threads per 1D block   
  dim3 dim1dBlock(TPB, 1, 1);

  // Set up pointers for the location to copy from and to
  Real *tmp1;
  Real *tmp2;

  // allocate buffer to copy conserved variable blocks to/from
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  Real *buffer;
  if (block_tot > 1) {
    if ( NULL == ( buffer = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real)) ) ) {
      printf("Failed to allocate CPU buffer.\n");
    }
    tmp1 = buffer;
    tmp2 = buffer;
  }
  else {
    tmp1 = host_conserved0;
    tmp2 = host_conserved1;
  }
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  buff += elapsedTime;
  #endif //TIME


  // allocate an array on the CPU to hold max_dti returned from each thread block
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  Real max_dti = 0;
  Real *host_dti_array;
  host_dti_array = (Real *) malloc(ngrid*sizeof(Real));
  #ifdef COOLING_GPU
  Real min_dt = 1e10;
  Real *host_dt_array;
  host_dt_array = (Real *) malloc(ngrid*sizeof(Real));
  #endif  
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  arrays += elapsedTime;
  #endif //TIME

  // allocate GPU arrays
  // conserved variables
  Real *dev_conserved, *dev_conserved_half;
  // input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
  #ifdef H_CORRECTION
  // arrays to hold the eta values for the H correction
  Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
  #endif
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;
  #ifdef COOLING_GPU
  // array of timesteps for dt calculation (cooling restriction)
  Real *dev_dt_array;
  #endif  

  // allocate memory on the GPU
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_conserved_half, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Lx,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Rx,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ly,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ry,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Lz,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Rz,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_y,   n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_z,   n_fields*BLOCK_VOL*sizeof(Real)) );
  #ifdef H_CORRECTION
  CudaSafeCall( cudaMalloc((void**)&eta_x,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_y,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_z,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_x, BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_y, BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_z, BLOCK_VOL*sizeof(Real)) );
  #endif //H_CORRECTION
  CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );
  #ifdef COOLING_GPU
  CudaSafeCall( cudaMalloc((void**)&dev_dt_array, ngrid*sizeof(Real)) );
  #endif  
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  arrays += elapsedTime;
  #endif //TIME


  // START LOOP OVER SUBGRID BLOCKS
  while (block < block_tot) {

    // copy the conserved variable block to the buffer
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    host_copy_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved0, buffer, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CPU copying: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    buff += elapsedTime;
    #endif //TIME    

    // calculate the global x, y, and z offsets of this subgrid block
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME
    get_offsets_3D(nx_s, ny_s, nz_s, n_ghost, x_off, y_off, z_off, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, &x_off_s, &y_off_s, &z_off_s);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Getting offsets: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    other += elapsedTime;
    #endif //TIME

    // copy the conserved variables onto the GPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME
    CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy to GPU: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    cpto += elapsedTime;
    #endif //TIME
 

    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME
    // Step 1: Use PCM reconstruction to put primitive variables into interface arrays
    PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("PCM reconstruction: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    pcm += elapsedTime;
    #endif //TIME


    // Step 2: Calculate first-order upwind fluxes 
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2, n_fields);
    #endif //ROE
    #ifdef HLLC 
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("x1 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r1x += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("y1 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r1y += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("z1 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r1z += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    #endif //HLLC
    CudaCheckError();


    // Step 3: Update the conserved variables half a timestep 
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME   
    Update_Conserved_Variables_3D_half<<<dim1dGrid,dim1dBlock>>>(dev_conserved, dev_conserved_half, F_x, F_y, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, 0.5*dt, gama, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Half step update: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    ie += elapsedTime;
    #endif //TIME 

    // Step 4: Construct left and right interface values using updated conserved variables
    #ifdef PCM
    PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    #endif
    #ifdef PLMP
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PLMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PLMP 
    #ifdef PLMC
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PLMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);  
    #endif
    #ifdef PPMP
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    PPMP_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    #endif //PPMP
    #ifdef PPMC
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("x ppm: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    ppmx += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("y ppm: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    ppmy += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    PPMC_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2, n_fields);
    CudaCheckError();
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("z ppm: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    ppmz += elapsedTime;
    #endif //TIME    
    #endif //PPMC
    CudaCheckError();
    

    #ifdef H_CORRECTION
    // Step 4.5: Calculate eta values for H correction
    calc_eta_x_3D<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, eta_x, nx_s, ny_s, nz_s, n_ghost, gama);
    calc_eta_y_3D<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, eta_y, nx_s, ny_s, nz_s, n_ghost, gama);
    calc_eta_z_3D<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, eta_z, nx_s, ny_s, nz_s, n_ghost, gama);
    CudaCheckError();
    // and etah values for each interface
    calc_etah_x_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_x, nx_s, ny_s, nz_s, n_ghost);
    calc_etah_y_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_y, nx_s, ny_s, nz_s, n_ghost);
    calc_etah_z_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_z, nx_s, ny_s, nz_s, n_ghost);
    CudaCheckError();
    #endif //H_CORRECTION


    // Step 5: Calculate the fluxes again
    #ifdef EXACT
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    Calculate_Exact_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #endif //EXACT
    #ifdef ROE
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1, n_fields);
    Calculate_Roe_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2, n_fields);
    #endif //ROE
    #ifdef HLLC 
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("x2 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r2x += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("y2 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r2y += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calculate_HLLC_Fluxes_CUDA<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("z2 fluxes: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    r2z += elapsedTime;
    cudaEventRecord(start, 0);
    #endif //TIME 
    #endif //HLLC
    CudaCheckError();


    // Step 6: Update the conserved variable array
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    Update_Conserved_Variables_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, F_x, F_y, F_z, nx_s, ny_s, nz_s, x_off_s, y_off_s, z_off_s, n_ghost, dx, dy, dz, xbound, ybound, zbound, dt, gama, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("conserved variable update: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    cvu += elapsedTime;
    #endif //TIME     
    CudaCheckError();

    #ifdef DE
    Sync_Energies_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama, n_fields);
    CudaCheckError();
    #endif

    // Apply cooling
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    #ifdef COOLING_GPU
    cooling_kernel<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, n_fields, dt, gama, dev_dt_array);  
    CudaCheckError();
    #endif
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    //printf("conserved variable update: %5.3f ms\n", elapsedTime);
    cool += elapsedTime;
    #endif //TIME     
 
    // Step 7: Calculate the next time step
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    Calc_dt_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dev_dti_array, gama);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("dti calc: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    dti += elapsedTime;
    #endif //TIME  
    CudaCheckError();

    // copy the updated conserved variable array back to the CPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME    
    CudaSafeCall( cudaMemcpy(tmp2, dev_conserved, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Copy from GPU: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    cpfr += elapsedTime;
    #endif //TIME    

    // copy the updated conserved variable array from the buffer into the host_conserved array on the CPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    host_return_block_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved1, buffer, n_fields);
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CPU copying: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    buff += elapsedTime;
    #endif //TIME    

    // copy the dti array onto the CPU
    #ifdef TIME
    cudaEventRecord(start, 0);
    #endif //TIME 
    CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
    // find maximum inverse timestep from CFL condition
    for (int i=0; i<ngrid; i++) {
      max_dti = fmax(max_dti, host_dti_array[i]);
    }
    #ifdef TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("dti copying & calc: %5.3f ms\n", elapsedTime);
    fflush(stdout);
    dti += elapsedTime;
    #endif //TIME  
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

  // free CPU memory
  free(host_dti_array);  
  if (block_tot > 1) free(buffer);
  #ifdef COOLING_GPU
  free(host_dt_array);  
  #endif  

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(dev_conserved_half);
  cudaFree(Q_Lx);
  cudaFree(Q_Rx);
  cudaFree(Q_Ly);
  cudaFree(Q_Ry);
  cudaFree(Q_Lz);
  cudaFree(Q_Rz);
  cudaFree(F_x);
  cudaFree(F_y);
  cudaFree(F_z);
  #ifdef H_CORRECTION
  cudaFree(eta_x);
  cudaFree(eta_y);
  cudaFree(eta_z);
  cudaFree(etah_x);
  cudaFree(etah_y);
  cudaFree(etah_z);
  #endif //H_CORRECTION
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  cudaFree(dev_dt_array);
  #endif


  #ifdef TIME
  //printf("cpto: %6.2f  cpfr: %6.2f\n", cpto, cpfr);
  //printf("ppmx: %6.2f  ppmy: %6.2f  ppmz: %6.2f\n", ppmx, ppmy, ppmz);
  //printf("r1x:  %6.2f  r1y:  %6.2f  r1z:  %6.2f\n", r1x, r1y, r1z);
  //printf("r2x:  %6.2f  r2y:  %6.2f  r2z:  %6.2f\n", r2x, r2y, r2z);
  //printf("pcm:  %6.2f  ie:   %6.2f  cvu:  %6.2f\n", pcm, ie, cvu);
  //printf("buff: %6.2f  dti:  %6.2f\n", buff, dti);
  //printf("cool: %6.2f\n", cool);
  //printf("arrs: %6.2f  othr: %6.2f\n", arrays, other);
  #endif
  #ifdef TIME
  cudaEventRecord(stop_VL, 0);
  cudaEventSynchronize(stop_VL);
  cudaEventElapsedTime(&elapsedTime, start_VL, stop_VL);
  printf("Total time for VL step: %5.3f ms\n", elapsedTime);
  #endif //TIME

  // return the maximum inverse timestep
  return max_dti;

}


__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma, int n_fields)
{
  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  int n_cells = nx*ny*nz;

  // get a global thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;
  int id = xid + yid*nx + zid*nx*ny;

  int imo = xid-1 + yid*nx + zid*nx*ny;
  int jmo = xid + (yid-1)*nx + zid*nx*ny;
  int kmo = xid + yid*nx + (zid-1)*nx*ny;

  #ifdef DE
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo, P;
  int ipo, jpo, kpo;
  #endif

  // threads corresponding to all cells except outer ring of ghost cells do the calculation
  if (xid > 0 && xid < nx-1 && yid > 0 && yid < ny-1 && zid > 0 && zid < nz-1)
  {
    #ifdef DE
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before half step update.\n");
    //if (P < 0.0) printf("%d Negative pressure before half step update.\n", id);
    ipo = xid+1 + yid*nx + zid*nx*ny;
    jpo = xid + (yid+1)*nx + zid*nx*ny;
    kpo = xid + yid*nx + (zid+1)*nx*ny;
    vx_imo = dev_conserved[1*n_cells + imo] / dev_conserved[imo]; 
    vx_ipo = dev_conserved[1*n_cells + ipo] / dev_conserved[ipo]; 
    vy_jmo = dev_conserved[2*n_cells + jmo] / dev_conserved[jmo]; 
    vy_jpo = dev_conserved[2*n_cells + jpo] / dev_conserved[jpo]; 
    vz_kmo = dev_conserved[3*n_cells + kmo] / dev_conserved[kmo]; 
    vz_kpo = dev_conserved[3*n_cells + kpo] / dev_conserved[kpo]; 
    #endif
  
    // update the conserved variable array
    dev_conserved_half[            id] = dev_conserved[            id]
                                       + dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                       + dtody * (dev_F_y[            jmo] - dev_F_y[            id])
                                       + dtodz * (dev_F_z[            kmo] - dev_F_z[            id]);
    dev_conserved_half[  n_cells + id] = dev_conserved[  n_cells + id] 
                                       + dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                       + dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                                       + dtodz * (dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_conserved_half[2*n_cells + id] = dev_conserved[2*n_cells + id] 
                                       + dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                       + dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                                       + dtodz * (dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_conserved_half[3*n_cells + id] = dev_conserved[3*n_cells + id] 
                                       + dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                       + dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                                       + dtodz * (dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_conserved_half[4*n_cells + id] = dev_conserved[4*n_cells + id] 
                                       + dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                       + dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                                       + dtodz * (dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    #ifdef SCALAR                                   
    for (int i=0; i<NSCALARS; i++) {
      dev_conserved_half[(5+i)*n_cells + id] = dev_conserved[(5+i)*n_cells + id] 
                                         + dtodx * (dev_F_x[(5+i)*n_cells + imo] - dev_F_x[(5+i)*n_cells + id])
                                         + dtody * (dev_F_y[(5+i)*n_cells + jmo] - dev_F_y[(5+i)*n_cells + id])
                                         + dtodz * (dev_F_z[(5+i)*n_cells + kmo] - dev_F_z[(5+i)*n_cells + id]);
    }                                   
    #endif
    #ifdef DE
    dev_conserved_half[(n_fields-1)*n_cells + id] = dev_conserved[(n_fields-1)*n_cells + id] 
                                       + dtodx * (dev_F_x[(n_fields-1)*n_cells + imo] - dev_F_x[(n_fields-1)*n_cells + id])
                                       + dtody * (dev_F_y[(n_fields-1)*n_cells + jmo] - dev_F_y[(n_fields-1)*n_cells + id])
                                       + dtodz * (dev_F_z[(n_fields-1)*n_cells + kmo] - dev_F_z[(n_fields-1)*n_cells + id])
                                       + 0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));
    #endif
    //if (dev_conserved_half[id] < 0.0 || dev_conserved_half[id] != dev_conserved_half[id] || dev_conserved_half[4*n_cells+id] < 0.0 || dev_conserved_half[4*n_cells+id] != dev_conserved_half[4*n_cells+id]) {
      //printf("%3d %3d %3d Thread crashed in half step update. d: %e E: %e\n", xid, yid, zid, dev_conserved_half[id], dev_conserved_half[4*n_cells+id]);
    //}    

  }

}




#endif //VL
#endif //CUDA
