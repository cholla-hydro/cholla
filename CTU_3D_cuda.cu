/*! \file CTU_3D_cuda.cu
 *  \brief Definitions of the cuda 3D CTU algorithm functions. */

#ifdef CUDA

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include"global.h"
#include"global_cuda.h"
#include"CTU_3D_cuda.h"
#include"pcm_cuda.h"
#include"plmp_ctu_cuda.h"
#include"plmc_ctu_cuda.h"
#include"ppmp_ctu_cuda.h"
#include"ppmc_ctu_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"h_correction_3D_cuda.h"
#include"cooling.h"
#include"subgrid_routines_3D.h"


//#define TIME 
//#define TURBULENCE


__global__ void test_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int block);


__global__ void Evolve_Interface_States_3D(Real *dev_conserved, Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           Real *dev_Q_Lz, Real *dev_Q_Rz, Real *dev_F_z,
                                           int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt);

__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma);

__global__ void calc_dt_cuda(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real *dti_array, Real gamma);


Real CTU_Algorithm_3D_CUDA(Real *host_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt)
{
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid
  //concatenated into a 1-d array

  #ifdef TIME
  // capture the start time
  cudaEvent_t start_CTU, stop_CTU;
  cudaEvent_t start, stop;
  cudaEventCreate(&start_CTU);
  cudaEventCreate(&stop_CTU);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start_CTU, 0);
  float elapsedTime;
  Real cpto, cpfr;
  Real buff, dti;
  Real ppmx, ppmy, ppmz;
  Real r1x, r1y, r1z;
  Real r2x, r2y, r2z;
  Real ie, cvu;
  cpto = cpfr = 0;
  buff = dti = 0;
  ppmx = ppmy = ppmz = 0;
  r1x = r1y = r1z = 0;
  r2x = r2y = r2z = 0;
  ie = cvu = 0;
  #endif

  int n_fields = 5;
  #ifdef DE
  n_fields = 6;
  #endif

  // dimensions of subgrid blocks
  int nx_s; //number of cells in the subgrid block along x direction
  int ny_s; //number of cells in the subgrid block along y direction
  int nz_s; //number of cells in the subgrid block along z direction

  // total number of blocks needed
  int block_tot;    //total number of subgrid blocks (unsplit == 1)
  int block1_tot;   //total number of subgrid blocks in x direction
  int block2_tot;   //total number of subgrid blocks in y direction
  int block3_tot;   //total number of subgrid blocks in z direction 
  int remainder1;   //modulus of number of cells after block subdivision in x direction
  int remainder2;   //modulus of number of cells after block subdivision in y direction 
  int remainder3;   //modulus of number of cells after block subdivision in z direction

  // counter for which block we're on
  int block = 0;

   
  // calculate the dimensions for each subgrid block
  sub_dimensions_3D(nx, ny, nz, n_ghost, &nx_s, &ny_s, &nz_s, &block1_tot, &block2_tot, &block3_tot, &remainder1, &remainder2, &remainder3);
  //printf("%d %d %d %d %d %d %d %d %d\n", nx_s, ny_s, nz_s, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3);
  block_tot = block1_tot*block2_tot*block3_tot;

  // number of cells in one subgrid block
  int BLOCK_VOL = nx_s*ny_s*nz_s;

  // define the dimensions for the 1D grid
  //int  ngrid = (n_cells + TPB - 1) / TPB;
  int  ngrid = (BLOCK_VOL + TPB - 1) / TPB;

  //number of blocks per 1-d grid  
  dim3 dim1dGrid(ngrid, 1, 1);

  //number of threads per 1-d block   
  dim3 dim1dBlock(TPB, 1, 1);


  // allocate buffer arrays to copy conserved variable slices into
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  Real **buffer;
  allocate_buffers_3D(block1_tot, block2_tot, block3_tot, BLOCK_VOL, buffer);
  // and set up pointers for the location to copy from and to
  Real *tmp1;
  Real *tmp2;
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  buff += elapsedTime;
  #endif //TIME


  // allocate an array on the CPU to hold max_dti returned from each thread block
  Real max_dti = 0;
  Real *host_dti_array;
  //printf("ngrid: %d\n", ngrid);
  host_dti_array = (Real *) malloc(ngrid*sizeof(Real));

  // allocate GPU arrays
  // conserved variables
  Real *dev_conserved;
  // input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
  // arrays to hold the eta values for the H correction
  Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;


  // allocate memory on the GPU
  CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Lx,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Rx,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ly,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Ry,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Lz,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_Rz,  n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_x,   n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_y,   n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F_z,   n_fields*BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_x,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_y,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&eta_z,  BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_x, BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_y, BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah_z, BLOCK_VOL*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );


  // transfer first conserved variable slice into the first buffer
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  host_copy_init_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, &tmp1, &tmp2);
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  buff += elapsedTime;
  #endif //TIME


  // START LOOP OVER SUBGRID BLOCKS HERE
  while (block < block_tot) {

  // zero the GPU arrays
  cudaMemset(dev_conserved, 0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Lx,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Rx,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Ly,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Ry,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Lz,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(Q_Rz,  0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(F_x,   0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(F_y,   0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(F_z,   0, n_fields*BLOCK_VOL*sizeof(Real));
  cudaMemset(eta_x,  0, BLOCK_VOL*sizeof(Real));
  cudaMemset(eta_y,  0, BLOCK_VOL*sizeof(Real));
  cudaMemset(eta_z,  0, BLOCK_VOL*sizeof(Real));
  cudaMemset(etah_x, 0, BLOCK_VOL*sizeof(Real));
  cudaMemset(etah_y, 0, BLOCK_VOL*sizeof(Real));
  cudaMemset(etah_z, 0, BLOCK_VOL*sizeof(Real));
  cudaMemset(dev_dti_array, 0, ngrid*sizeof(Real));  
  CudaCheckError();


  // copy the conserved variables onto the GPU
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  CudaSafeCall( cudaMemcpy(dev_conserved, tmp1, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyHostToDevice) );
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("GPU copy: %5.3f ms\n", elapsedTime);
  cpto += elapsedTime;
  #endif //TIME
  

  // Step 1: Do the reconstruction
  #ifdef PCM
  PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama);
  CudaCheckError();
  #endif //PCM
  #ifdef PLMP
  PLMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
  PLMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
  PLMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2);
  CudaCheckError();
  #endif //PLMP 
  #ifdef PLMC
  PLMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
  PLMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
  PLMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2);
  CudaCheckError();
  #endif //PLMC 
  #ifdef PPMP
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  PPMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("x ppm: %5.3f ms\n", elapsedTime);
  ppmx += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME  
  PPMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
  CudaCheckError();
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("y ppm: %5.3f ms\n", elapsedTime);
  ppmy += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME
  PPMP_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("z ppm: %5.3f ms\n", elapsedTime);
  ppmz += elapsedTime;
  #endif //TIME 
  #endif //PPMP
  #ifdef PPMC
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  PPMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("x ppm: %5.3f ms\n", elapsedTime);
  ppmx += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME
  PPMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, dy, dt, gama, 1);
  CudaCheckError();
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("y ppm: %5.3f ms\n", elapsedTime);
  ppmy += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME
  PPMC_CTU<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, dz, dt, gama, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("z ppm: %5.3f ms\n", elapsedTime);
  ppmz += elapsedTime;
  #endif //TIME 
  #endif //PPMC


  #ifdef COOLING
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Lx, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Ly, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Lz, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Rx, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Ry, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(Q_Rz, nx_s, ny_s, nz_s, n_ghost, 0.5*dt, gama);
  #endif



  // Step 2: Calculate the fluxes
  #ifdef EXACT
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("x fluxes: %5.3f ms\n", elapsedTime);
  r1x += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("y fluxes: %5.3f ms\n", elapsedTime);
  r1y += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("z fluxes: %5.3f ms\n", elapsedTime);
  r1z += elapsedTime;
  #endif //TIME
  #endif //EXACT

  #ifdef ROE
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("x fluxes: %5.3f ms\n", elapsedTime);
  r1x += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("y fluxes: %5.3f ms\n", elapsedTime);
  r1y += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("z fluxes: %5.3f ms\n", elapsedTime);
  r1z += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME    
  #endif //ROE

/*
  // Step 3: Evolve the interface states
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME   
  Evolve_Interface_States_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, F_x, Q_Ly, Q_Ry, F_y, Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dt);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("interface evolution: %5.3f ms\n", elapsedTime);
  ie += elapsedTime;
  #endif //TIME    
   

  #ifdef H_CORRECTION
  // Step 3.5: Calculate eta values for H correction
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME     
  calc_eta_x_3D<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, eta_x, nx_s, ny_s, nz_s, n_ghost, gama);
  CudaCheckError();
  calc_eta_y_3D<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, eta_y, nx_s, ny_s, nz_s, n_ghost, gama);
  CudaCheckError();
  calc_eta_z_3D<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, eta_z, nx_s, ny_s, nz_s, n_ghost, gama);
  CudaCheckError();
  // and etah values for each interface
  calc_etah_x_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_x, nx_s, ny_s, nz_s, n_ghost);
  CudaCheckError();
  calc_etah_y_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_y, nx_s, ny_s, nz_s, n_ghost);
  CudaCheckError();
  calc_etah_z_3D<<<dim1dGrid,dim1dBlock>>>(eta_x, eta_y, eta_z, etah_z, nx_s, ny_s, nz_s, n_ghost);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("H correction: %5.3f ms\n", elapsedTime);
  #endif //TIME 
  #endif //H_CORRECTION


  // Step 4: Calculate the fluxes again
  #ifdef EXACT
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("x fluxes: %5.3f ms\n", elapsedTime);
  r2x += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME   
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, 1);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("y fluxes: %5.3f ms\n", elapsedTime);
  r2y += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME   
  Calculate_Exact_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("z fluxes: %5.3f ms\n", elapsedTime);
  r2z += elapsedTime;
  #endif //TIME
  #endif //EXACT

  #ifdef ROE
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("x fluxes: %5.3f ms\n", elapsedTime);
  r2x += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("y fluxes: %5.3f ms\n", elapsedTime);
  r2y += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME 
  Calculate_Roe_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("z fluxes: %5.3f ms\n", elapsedTime);
  r2z += elapsedTime;
  #endif //TIME 
  #endif //ROE
*/

  // Step 5: Update the conserved variable array
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME   
  Update_Conserved_Variables_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, F_x, F_y, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dt, gama);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("conserved variable update: %5.3f ms\n", elapsedTime);
  cvu += elapsedTime;
  #endif //TIME     

  #ifdef COOLING
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dt, gama);
  #endif

  // Step 6: Calculate the next timestep
  calc_dt_cuda<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dev_dti_array, gama);
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
  //printf("GPU return: %5.3f ms\n", elapsedTime);
  cpfr += elapsedTime;
  #endif //TIME    


  // copy the next conserved variable blocks into appropriate buffers
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME 
  host_copy_next_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved, buffer, &tmp1);

  // copy the updated conserved variable array back into the host_conserved array on the CPU
  host_return_values_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved, buffer);
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("CPU copying: %5.3f ms\n", elapsedTime);
  buff += elapsedTime;
  #endif //TIME    


  // copy the dti array onto the CPU
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME 
  CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // iterate through to find the maximum inverse dt for this subgrid block
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, host_dti_array[i]);
  }
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("dti copying & calc: %5.3f ms\n", elapsedTime);
  dti += elapsedTime;
  #endif //TIME     


  // add one to the counter
  block++;

}


  // free CPU memory
  free(host_dti_array);  
  free_buffers_3D(nx, ny, nz, nx_s, ny_s, nz_s, block1_tot, block2_tot, block3_tot, buffer);


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
  cudaFree(eta_x);
  cudaFree(eta_y);
  cudaFree(eta_z);
  cudaFree(etah_x);
  cudaFree(etah_y);
  cudaFree(etah_z);
  cudaFree(dev_dti_array);

  #ifdef TIME
  //printf("cpto: %6.2f  cpfr: %6.2f\n", cpto, cpfr);
  //printf("ppmx: %6.2f  ppmy: %6.2f  ppmz: %6.2f\n", ppmx, ppmy, ppmz);
  //printf("r1x:  %6.2f  r1y:  %6.2f  r1z:  %6.2f\n", r1x, r1y, r1z);
  //printf("r2x:  %6.2f  r2y:  %6.2f  r2z:  %6.2f\n", r2x, r2y, r2z);
  //printf("ie:   %6.2f  cvu:  %6.2f\n", ie, cvu);
  //printf("buff: %6.2f  dti:  %6.2f\n", buff, dti);
  #endif

  #ifdef TIME
  cudaEventRecord(stop_CTU, 0);
  cudaEventSynchronize(stop_CTU);
  cudaEventElapsedTime(&elapsedTime, start_CTU, stop_CTU);
  //printf("Time for CTU step: %5.3f ms\n", elapsedTime);
  #endif //TIME

  #ifdef TIME
  cudaEventDestroy(start_CTU);
  cudaEventDestroy(stop_CTU);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  #endif


  // return the maximum inverse timestep
  return max_dti;

}


__global__ void test_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int block)
{

  // get a thread ID
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = tid / (nx*ny);
  int yid = (tid - zid*nx*ny) / nx;
  int xid = tid - zid*nx*ny - yid*nx;

  // assign each real cell the block number 
  if (xid >= n_ghost && xid < nx-n_ghost && yid >= n_ghost && yid < ny-n_ghost && zid >= n_ghost && zid < nz-n_ghost) {
    dev_conserved[tid] = block;
  }


}


__global__ void Evolve_Interface_States_3D(Real *dev_conserved, Real *dev_Q_Lx, Real *dev_Q_Rx, Real *dev_F_x,
                                           Real *dev_Q_Ly, Real *dev_Q_Ry, Real *dev_F_y,
                                           Real *dev_Q_Lz, Real *dev_Q_Rz, Real *dev_F_z,
                                           int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt)
{
  Real d, d_inv, vx, vy, vz, P;
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
    d  =  dev_Q_Lx[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Lx[1*n_cells + id] * d_inv;
    vy = dev_Q_Lx[2*n_cells + id] * d_inv;
    vz = dev_Q_Lx[3*n_cells + id] * d_inv;
    P  = dev_Q_Lx[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Lx update. %f %f %f %f\n", xid, yid, zid, dev_Q_Lx[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Lx[id] < 0.0 || dev_Q_Lx[id] != dev_Q_Lx[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Lx update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Lx[id], dev_F_y[jmo], dev_F_y[id], dev_F_z[kmo], dev_F_z[id]);
    }
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
    d  =  dev_Q_Rx[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Rx[1*n_cells + id] * d_inv;
    vy = dev_Q_Rx[2*n_cells + id] * d_inv;
    vz = dev_Q_Rx[3*n_cells + id] * d_inv;
    P  = dev_Q_Rx[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Rx update. %f %f %f %f\n", xid, yid, zid, dev_Q_Rx[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Rx[id] < 0.0 || dev_Q_Rx[id] != dev_Q_Rx[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Rx update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Rx[id], dev_F_y[ipojmo], dev_F_y[ipo], dev_F_z[ipokmo], dev_F_z[ipo]);
    }
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
    d  =  dev_Q_Ly[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Ly[1*n_cells + id] * d_inv;
    vy = dev_Q_Ly[2*n_cells + id] * d_inv;
    vz = dev_Q_Ly[3*n_cells + id] * d_inv;
    P  = dev_Q_Ly[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Ly update. %f %f %f %f\n", xid, yid, zid, dev_Q_Ly[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Ly[id] < 0.0 || dev_Q_Ly[id] != dev_Q_Ly[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Ly update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Ly[id], dev_F_z[kmo], dev_F_z[id], dev_F_x[imo], dev_F_x[id]);
    }
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
    d  =  dev_Q_Ry[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Ry[1*n_cells + id] * d_inv;
    vy = dev_Q_Ry[2*n_cells + id] * d_inv;
    vz = dev_Q_Ry[3*n_cells + id] * d_inv;
    P  = dev_Q_Ry[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Ry update. %f %f %f %f\n", xid, yid, zid, dev_Q_Ry[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Ry[id] < 0.0 || dev_Q_Ry[id] != dev_Q_Ry[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Ry update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Ry[id], dev_F_z[jpokmo], dev_F_z[jpo], dev_F_x[jpoimo], dev_F_x[jpo]);
    }
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
    d  = dev_Q_Lz[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Lz[1*n_cells + id] * d_inv;
    vy = dev_Q_Lz[2*n_cells + id] * d_inv;
    vz = dev_Q_Lz[3*n_cells + id] * d_inv;
    P  = dev_Q_Lz[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Lz update. %f %f %f %f\n", xid, yid, zid, dev_Q_Lz[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Lz[id] < 0.0 || dev_Q_Lz[id] != dev_Q_Lz[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Lz update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Lz[id], dev_F_x[imo], dev_F_x[id], dev_F_y[jmo], dev_F_y[id]);
    }
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
    d  = dev_Q_Rz[            id];
    d_inv = 1.0 / d;
    vx = dev_Q_Rz[1*n_cells + id] * d_inv;
    vy = dev_Q_Rz[2*n_cells + id] * d_inv;
    vz = dev_Q_Rz[3*n_cells + id] * d_inv;
    P  = dev_Q_Rz[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure in Q_Rz update. %f %f %f %f\n", xid, yid, zid, dev_Q_Rz[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    if (dev_Q_Rz[id] < 0.0 || dev_Q_Rz[id] != dev_Q_Rz[id]) {
      printf("%3d %3d %3d Thread crashed in Q_Rz update. %f %f %f %f %f\n", xid, yid, zid, dev_Q_Rz[id], dev_F_x[kpoimo], dev_F_x[kpo], dev_F_y[kpojmo], dev_F_y[kpo]);
    }
  }

}


__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt,
                                              Real gamma)
{
  Real d, d_inv, vx, vy, vz, P;
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;

  #ifdef DE
  Real ge1, ge2, xvmax, yvmax, zvmax, vmax;
  int ipo, jpo, kpo;
  #endif

  Real dtodx = dt/dx;
  Real dtody = dt/dy;
  Real dtodz = dt/dz;
  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;


  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    //if (d < 0.0 || d != d) printf("Negative density before final update.\n");
    //if (P < 0.0) printf("%d Negative pressure before final update.\n", id);

    // update the conserved variable array
    imo = xid-1 + yid*nx + zid*nx*ny;
    jmo = xid + (yid-1)*nx + zid*nx*ny;
    kmo = xid + yid*nx + (zid-1)*nx*ny;
    dev_conserved[            id] += dtodx * (dev_F_x[            imo] - dev_F_x[            id])
                                  +  dtody * (dev_F_y[            jmo] - dev_F_y[            id])
                                  +  dtodz * (dev_F_z[            kmo] - dev_F_z[            id]);
    dev_conserved[  n_cells + id] += dtodx * (dev_F_x[  n_cells + imo] - dev_F_x[  n_cells + id])
                                  +  dtody * (dev_F_y[  n_cells + jmo] - dev_F_y[  n_cells + id])
                                  +  dtodz * (dev_F_z[  n_cells + kmo] - dev_F_z[  n_cells + id]);
    dev_conserved[2*n_cells + id] += dtodx * (dev_F_x[2*n_cells + imo] - dev_F_x[2*n_cells + id])
                                  +  dtody * (dev_F_y[2*n_cells + jmo] - dev_F_y[2*n_cells + id])
                                  +  dtodz * (dev_F_z[2*n_cells + kmo] - dev_F_z[2*n_cells + id]);
    dev_conserved[3*n_cells + id] += dtodx * (dev_F_x[3*n_cells + imo] - dev_F_x[3*n_cells + id])
                                  +  dtody * (dev_F_y[3*n_cells + jmo] - dev_F_y[3*n_cells + id])
                                  +  dtodz * (dev_F_z[3*n_cells + kmo] - dev_F_z[3*n_cells + id]);
    dev_conserved[4*n_cells + id] += dtodx * (dev_F_x[4*n_cells + imo] - dev_F_x[4*n_cells + id])
                                  +  dtody * (dev_F_y[4*n_cells + jmo] - dev_F_y[4*n_cells + id])
                                  +  dtodz * (dev_F_z[4*n_cells + kmo] - dev_F_z[4*n_cells + id]);
    #ifdef DE
    dev_conserved[5*n_cells + id] += (dtodx * (dev_F_x[5*n_cells + imo] - dev_F_x[5*n_cells + id])
                                  +   dtody * (dev_F_y[5*n_cells + jmo] - dev_F_y[5*n_cells + id])
                                  +   dtodz * (dev_F_z[5*n_cells + kmo] - dev_F_z[5*n_cells + id])
                                  +   P * (dtodx*vx + dtody*vy + dtodz*vz)) / dev_conserved[id];
    #endif
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id]) {
      printf("%3d %3d %3d Thread crashed in final update. %f %f %f %f %f %f\n", xid, yid, zid, d, dtodx*(dev_F_x[imo]-dev_F_x[id]), dev_F_y[jmo],dev_F_y[id], dtodz*(dev_F_z[kmo]-dev_F_z[id]), dev_conserved[id]);
    }
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure after final update. %f %f %f %f %f\n", xid, yid, zid, dev_conserved[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz, P);
    }
    #ifdef DE
    syncthreads();
    // separately tracked specific internal energy 
    ge1 =  dev_conserved[5*n_cells + id];
    // specific internal energy calculated from total energy
    ge2 = dev_conserved[4*n_cells + id] * d_inv - 0.5*(vx*vx + vy*vy + vz*vz);
    if (ge1 < 0.0) printf("%d %d %d Negative internal energy after final update. %f %f %f\n", xid, yid, zid, dev_F_x[5*n_cells+imo] - dev_F_x[5*n_cells+id], dev_F_y[5*n_cells+jmo] - dev_F_y[5*n_cells+id], dev_F_z[5*n_cells+kmo] - dev_F_z[5*n_cells+id]); 
    //find the max nearby velocity difference 
    ipo = min(xid+1, nx-n_ghost-1);
    ipo = ipo + yid*nx + zid*nx*ny;
    jpo = min(yid+1, ny-n_ghost-1);
    jpo = xid + jpo*nx + zid*nx*ny;
    kpo = min(zid+1, nz-n_ghost-1);
    kpo = xid + yid*nx + kpo*nx*ny;
    xvmax = fmax(fabs(vx-dev_conserved[1*n_cells + imo]/dev_conserved[imo]), fabs(dev_conserved[1*n_cells + ipo]/dev_conserved[ipo] - vx));
    yvmax = fmax(fabs(vy-dev_conserved[2*n_cells + jmo]/dev_conserved[jmo]), fabs(dev_conserved[2*n_cells + jpo]/dev_conserved[jpo] - vy));
    zvmax = fmax(fabs(vz-dev_conserved[3*n_cells + kmo]/dev_conserved[kmo]), fabs(dev_conserved[3*n_cells + kpo]/dev_conserved[kpo] - vz));
    vmax = fmax(xvmax, yvmax);
    vmax = fmax(vmax, zvmax);
    // if the conservatively calculated internal energy is greater than an the estimate of the truncation error 
    // use the internal energy computed from the total energy to do the internal energy update
    if (ge2 > 0.25*vmax*vmax) {
      dev_conserved[5*n_cells + id] = ge2;
      ge1 = ge2;
    }
    // update the total energy
    dev_conserved[4*n_cells + id] += d*ge1 - d*ge2; 
    #endif
  }

}


__global__ void calc_dt_cuda(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real *dti_array, Real gamma)
{
  __shared__ Real max_dti[TPB];

  Real d, d_inv, vx, vy, vz, P, cs;
  int id, xid, yid, zid, n_cells;
  int tid;

  n_cells = nx*ny*nz;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx*ny);
  yid = (id - zid*nx*ny) / nx;
  xid = id - zid*nx*ny - yid*nx;
  // and a thread id within the block  
  tid = threadIdx.x;

  // set shared memory to 0
  max_dti[tid] = 0;
  __syncthreads();

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost-1 && xid < nx-n_ghost && yid > n_ghost-1 && yid < ny-n_ghost && zid > n_ghost-1 && zid < nz-n_ghost)
  {
    // every thread collects the conserved variables it needs from global memory
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    cs = sqrt(d_inv * gamma * P);
    //max_dti[tid] = fmax((fabs(vx)+cs)/dx, (fabs(vy)+cs)/dy);
    //max_dti[tid] = fmax(max_dti[tid], (fabs(vz)+cs)/dz);
    max_dti[tid] = (fabs(vx)+cs)/dx + (fabs(vy)+cs)/dy + (fabs(vz)+cs)/dz;
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
  if (tid == 0) dti_array[blockIdx.x] = max_dti[0];

}


#endif //CUDA
