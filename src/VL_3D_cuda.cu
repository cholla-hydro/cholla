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
#include"plmp_vl_cuda.h"
#include"ppmp_vl_cuda.h"
#include"ppmc_vl_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"h_correction_3D_cuda.h"
#include"cooling_cuda.h"
#include"subgrid_routines_3D.h"


//#define TIME 
//#define TEST
//#define TURBULENCE

__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma);


Real VL_Algorithm_3D_CUDA(Real *host_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt)
{

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

  #ifdef COOLING_GPU
  // allocate CUDA arrays for cooling/heating tables
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray* cuCoolArray;
  cudaArray* cuHeatArray;
  cudaMallocArray(&cuCoolArray, &channelDesc, 81, 121);
  cudaMallocArray(&cuHeatArray, &channelDesc, 81, 121);
  // Copy to device memory the cooling and heating arrays
  // in host memory
  cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, 81*121*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToArray(cuHeatArray, 0, 0, heating_table, 81*121*sizeof(float), cudaMemcpyHostToDevice);

  // Specify textures
  struct cudaResourceDesc coolResDesc;
  memset(&coolResDesc, 0, sizeof(coolResDesc));
  coolResDesc.resType = cudaResourceTypeArray;
  coolResDesc.res.array.array = cuCoolArray;
  struct cudaResourceDesc heatResDesc;
  memset(&heatResDesc, 0, sizeof(heatResDesc));
  heatResDesc.resType = cudaResourceTypeArray;
  heatResDesc.res.array.array = cuHeatArray;  

  // Specify texture object parameters (same for both tables)
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture objects
  cudaTextureObject_t coolTexObj = 0;
  cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
  cudaTextureObject_t heatTexObj = 0;
  cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);
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
  sub_dimensions_3D(nx, ny, nz, n_ghost, &nx_s, &ny_s, &nz_s, &block1_tot, &block2_tot, &block3_tot, &remainder1, &remainder2, &remainder3, n_fields);
  block_tot = block1_tot*block2_tot*block3_tot;

  // number of cells in one subgrid block
  int BLOCK_VOL = nx_s*ny_s*nz_s;


  // define the dimensions for the 1D grid
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
  allocate_buffers_3D(block1_tot, block2_tot, block3_tot, BLOCK_VOL, buffer, n_fields);
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
  host_dti_array = (Real *) malloc(ngrid*sizeof(Real));

  // allocate GPU arrays
  // conserved variables
  Real *dev_conserved, *dev_conserved_half;
  // input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
  // arrays to hold the eta values for the H correction
  Real *eta_x, *eta_y, *eta_z, *etah_x, *etah_y, *etah_z;
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;

  #ifdef TEST
  Real *test1;
  Real *test2;
  test1 = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real));
  test2 = (Real *) malloc(n_fields*BLOCK_VOL*sizeof(Real));
  #endif

  // allocate memory on the GPU
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
  host_copy_init_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, remainder1, remainder2, BLOCK_VOL, host_conserved, buffer, &tmp1, &tmp2, n_fields);
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
  cudaMemset(dev_conserved_half, 0, n_fields*BLOCK_VOL*sizeof(Real));
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
  

  // Step 1: Use PCM reconstruction to put primitive variables into interface arrays
  PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama);
  CudaCheckError();

  #ifdef TEST
  CudaSafeCall( cudaMemcpy(test1, Q_Lx, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
  CudaSafeCall( cudaMemcpy(test2, Q_Rx, n_fields*BLOCK_VOL*sizeof(Real), cudaMemcpyDeviceToHost) );
  printf("After pcm\n");
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
        int id = i + j*nx + k*nx*ny;
        int idpo = i+1 + j*nx + k*nx*ny;
        printf("%3d %3d %3d %f %f %f %f\n", i, j, k, host_conserved[id], host_conserved[idpo], test1[id], test2[id]);
      }
    }
  }
  #endif

  // Step 2: Calculate first-order upwind fluxes 
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
  #ifdef HLLC
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2);
  CudaCheckError();
  #endif


  // Step 3: Update the conserved variables half a timestep 
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME   
  Update_Conserved_Variables_3D_half<<<dim1dGrid,dim1dBlock>>>(dev_conserved, dev_conserved_half, F_x, F_y, F_z, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, 0.5*dt, gama);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  //printf("interface evolution: %5.3f ms\n", elapsedTime);
  ie += elapsedTime;
  #endif //TIME    


  // Step 4: Construct left and right interface values using updated conserved variables
  #ifdef PCM
  PCM_Reconstruction_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, Q_Ly, Q_Ry, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama);
  #endif
  #ifdef PLMP
  PLMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
  CudaCheckError();
  PLMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
  CudaCheckError();
  PLMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, 2);
  CudaCheckError();
  #endif //PLMP 
  #ifdef PLMC
  printf("PLMC not supported with Van Leer integrator.\n");
  exit(0);
  #endif
  #ifdef PPMP
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif //TIME  
  PPMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("x ppm: %5.3f ms\n", elapsedTime);
  ppmx += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME    
  PPMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
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
  PPMP_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, 2);
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
  PPMC_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lx, Q_Rx, nx_s, ny_s, nz_s, n_ghost, gama, 0);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("x ppm: %5.3f ms\n", elapsedTime);
  ppmx += elapsedTime;
  cudaEventRecord(start, 0);
  #endif //TIME    
  PPMC_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Ly, Q_Ry, nx_s, ny_s, nz_s, n_ghost, gama, 1);
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
  PPMC_VL<<<dim1dGrid,dim1dBlock>>>(dev_conserved_half, Q_Lz, Q_Rz, nx_s, ny_s, nz_s, n_ghost, gama, 2);
  CudaCheckError();
  #ifdef TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("z ppm: %5.3f ms\n", elapsedTime);
  ppmz += elapsedTime;
  #endif //TIME   
  #endif //PPMC
  

  #ifdef H_CORRECTION
  // Step 4.5: Calculate eta values for H correction
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


  // Step 5: Calculate the fluxes again
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
  #ifdef HLLC
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lx, Q_Rx, F_x, nx_s, ny_s, nz_s, n_ghost, gama, etah_x, 0);
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Ly, Q_Ry, F_y, nx_s, ny_s, nz_s, n_ghost, gama, etah_y, 1);
  Calculate_HLLC_Fluxes<<<dim1dGrid,dim1dBlock>>>(Q_Lz, Q_Rz, F_z, nx_s, ny_s, nz_s, n_ghost, gama, etah_z, 2);
  CudaCheckError();
  #endif


  // Step 6: Update the conserved variable array
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

  #ifdef DE
  Sync_Energies_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, gama);
  #endif

  #ifdef COOLING_GPU
  cooling_kernel<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dt, gama, coolTexObj, heatTexObj);
  #endif

  
  // Step 7: Calculate the next time step
  Calc_dt_3D<<<dim1dGrid,dim1dBlock>>>(dev_conserved, nx_s, ny_s, nz_s, n_ghost, dx, dy, dz, dev_dti_array, gama);
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
  host_copy_next_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved, buffer, &tmp1, n_fields);

  // copy the updated conserved variable array back into the host_conserved array on the CPU
  host_return_values_3D(nx, ny, nz, nx_s, ny_s, nz_s, n_ghost, block, block1_tot, block2_tot, block3_tot, remainder1, remainder2, remainder3, BLOCK_VOL, host_conserved, buffer, n_fields);
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

  #ifdef TEST
  printf("Final values\n");
  for (int i=0; i<nx; i++) {
    for (int j=0; j<ny; j++) {
      for (int k=0; k<nz; k++) {
        int id = i + j*nx + k*nx*ny;
        int n_cells = nx*ny*nz;
        printf("%3d %3d %3d %f\n", i, j, k, host_conserved[id]);
      }
    }
  }
  #endif

  // free CPU memory
  free(host_dti_array);  
  free_buffers_3D(nx, ny, nz, nx_s, ny_s, nz_s, block1_tot, block2_tot, block3_tot, buffer);

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
  cudaFree(eta_x);
  cudaFree(eta_y);
  cudaFree(eta_z);
  cudaFree(etah_x);
  cudaFree(etah_y);
  cudaFree(etah_z);
  cudaFree(dev_dti_array);
  #ifdef COOLING_GPU
  // Destroy texture object
  cudaDestroyTextureObject(coolTexObj);
  cudaDestroyTextureObject(heatTexObj);
  // Free device memory
  cudaFreeArray(cuCoolArray);
  cudaFreeArray(cuHeatArray);  
  #endif

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
  
  #ifdef TEST
  free(test1);
  free(test2);
  #endif




  // return the maximum inverse timestep
  return max_dti;

}



__global__ void Update_Conserved_Variables_3D_half(Real *dev_conserved, Real *dev_conserved_half, Real *dev_F_x, Real *dev_F_y,  Real *dev_F_z,
                                              int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma)
{

  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;

  #ifdef DE
  Real d, d_inv, vx, vy, vz, P;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
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
  imo = xid-1 + yid*nx + zid*nx*ny;
  jmo = xid + (yid-1)*nx + zid*nx*ny;
  kmo = xid + yid*nx + (zid-1)*nx*ny;

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
    //if (d < 0.0 || d != d) printf("Negative density before final update.\n");
    //if (P < 0.0) printf("%d Negative pressure before final update.\n", id);
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
    #ifdef DE
    dev_conserved_half[5*n_cells + id] = dev_conserved[5*n_cells + id] 
                                       + dtodx * (dev_F_x[5*n_cells + imo] - dev_F_x[5*n_cells + id])
                                       + dtody * (dev_F_y[5*n_cells + jmo] - dev_F_y[5*n_cells + id])
                                       + dtodz * (dev_F_z[5*n_cells + kmo] - dev_F_z[5*n_cells + id])
                                       +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) + dtodz*(vz_kmo-vz_kpo));                                       
    #endif
    if (dev_conserved_half[id] < 0.0 || dev_conserved_half[id] != dev_conserved_half[id]) {
      //printf("%3d %3d %3d Thread crashed in half step update. %f %f %f %f\n", xid, yid, zid, dev_conserved[id], dtodx*(dev_F_x[imo]-dev_F_x[id]), dtody*(dev_F_y[jmo]-dev_F_y[id]), dtodz*(dev_F_z[kmo]-dev_F_z[id]));
    } 
    /*
    d  =  dev_conserved_half[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved_half[1*n_cells + id] * d_inv;
    vy =  dev_conserved_half[2*n_cells + id] * d_inv;
    vz =  dev_conserved_half[3*n_cells + id] * d_inv;
    P  = (dev_conserved_half[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) * (gamma - 1.0);
    if (P < 0.0) {
      printf("%3d %3d %3d Negative pressure after half step update. %f %f %f %f\n", xid, yid, zid, dev_conserved_half[4*n_cells + id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz);
    }
    */

  }

}



#endif //VL
#endif //CUDA
