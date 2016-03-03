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
#include"plmp_ctu_cuda.h"
#include"plmc_ctu_cuda.h"
#include"ppmp_ctu_cuda.h"
#include"ppmc_ctu_cuda.h"
#include"exact_cuda.h"
#include"roe_cuda.h"
#include"hllc_cuda.h"
#include"cooling_cuda.h"
#include"error_handling.h"
#include"io.h"


//#define TEST

Real CTU_Algorithm_1D_CUDA(Real *host_conserved, int nx, int n_ghost, Real dx, Real dt)
{
  //Here, *host_conserved contains the entire
  //set of conserved variables on the grid

  // capture the start time
  #ifdef TIME
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsedTime;
  #endif

  int n_cells = nx;
  int ny = 1;
  int nz = 1;

  int n_fields = 5;
  #ifdef DE
  n_fields = 6;
  #endif

  // set the dimensions of the cuda grid
  int  ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dimGrid(ngrid, 1, 1);
  dim3 dimBlock(TPB, 1, 1);

  // allocate an array on the CPU to hold max_dti returned from each thread block
  Real max_dti = 0;
  Real *host_dti_array;
  host_dti_array = (Real *) malloc(ngrid*sizeof(Real));

  #ifdef TEST
  Real *test1, *test2;
  test1 = (Real *) malloc(5*n_cells*sizeof(Real));
  test2 = (Real *) malloc(5*n_cells*sizeof(Real));
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

  // allocate GPU arrays
  // conserved variables
  Real *dev_conserved;
  // initial input states and associated interface fluxes (Q* and F* from Stone, 2008)
  Real *Q_L, *Q_R, *F;
  // array to hold zero values for H correction (necessary to pass to Roe solver)
  Real *etah;
  // array of inverse timesteps for dt calculation
  Real *dev_dti_array;


  // allocate memory on the GPU
  CudaSafeCall( cudaMalloc((void**)&dev_conserved, n_fields*n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_L, n_fields*n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&Q_R, n_fields*n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&F,   (n_fields)*n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&etah, n_cells*sizeof(Real)) );
  CudaSafeCall( cudaMalloc((void**)&dev_dti_array, ngrid*sizeof(Real)) );

  // zero all the GPU arrays
  cudaMemset(dev_conserved, 0, n_fields*n_cells*sizeof(Real));
  cudaMemset(Q_L, 0, n_fields*n_cells*sizeof(Real));
  cudaMemset(Q_R, 0, n_fields*n_cells*sizeof(Real));
  cudaMemset(F, 0, n_fields*n_cells*sizeof(Real));
  cudaMemset(etah, 0, n_cells*sizeof(Real));
  cudaMemset(dev_dti_array, 0, ngrid*sizeof(Real));
  CudaCheckError();


  // copy the conserved variable array onto the GPU
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif
  CudaSafeCall( cudaMemcpy(dev_conserved, host_conserved, n_fields*n_cells*sizeof(Real), cudaMemcpyHostToDevice) );
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU copy: %5.3f ms\n", elapsedTime);
  #endif
  CudaCheckError();


  // Step 1: Do the reconstruction
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif
  #ifdef PCM
  PCM_Reconstruction_1D<<<dimGrid,dimBlock>>>(dev_conserved, Q_L, Q_R, nx, n_ghost, gama);
  CudaCheckError();
  #endif
  #ifdef PLMP
  PLMP_CTU<<<dimGrid,dimBlock>>>(dev_conserved, Q_L, Q_R, nx, ny, nz, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #endif
  #ifdef PLMC
  PLMC_CTU<<<dimGrid,dimBlock>>>(dev_conserved, Q_L, Q_R, nx, ny, nz, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #endif
  #ifdef PPMP
  PPMP_CTU<<<dimGrid,dimBlock>>>(dev_conserved, Q_L, Q_R, nx, ny, nz, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #endif
  #ifdef PPMC
  PPMC_CTU<<<dimGrid,dimBlock>>>(dev_conserved, Q_L, Q_R, nx, ny, nz, n_ghost, dx, dt, gama, 0);
  CudaCheckError();
  #endif
  #ifdef TEST
  CudaSafeCall( cudaMemcpy(test1, Q_L, 5*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );  
  CudaSafeCall( cudaMemcpy(test2, Q_R, 5*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );  
  for (int i=0; i<nx; i++) {
    printf("%d %f %f\n", i, test1[i], test2[i]);
  }
  #endif

  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time to do reconstruction: %5.3f ms\n", elapsedTime);
  #endif

  
  // Step 2: Calculate the fluxes
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif
  #ifdef EXACT
  Calculate_Exact_Fluxes<<<dimGrid,dimBlock>>>(Q_L, Q_R, F, nx, ny, nz, n_ghost, gama, 0);
  #endif
  #ifdef ROE
  Calculate_Roe_Fluxes<<<dimGrid,dimBlock>>>(Q_L, Q_R, F, nx, ny, nz, n_ghost, gama, etah, 0);
  #endif
  #ifdef HLLC
  Calculate_HLLC_Fluxes<<<dimGrid,dimBlock>>>(Q_L, Q_R, F, nx, ny, nz, n_ghost, gama, etah, 0);
  #endif
  CudaCheckError();
  #ifdef TIME
  // get stop time, and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Time to do riemann problem:  %5.3f ms\n", elapsedTime);
  #endif


  // Step 3: Update the conserved variable array
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif      
  Update_Conserved_Variables_1D<<<dimGrid,dimBlock>>>(dev_conserved, F, n_cells, n_ghost, dx, dt, gama);
  CudaCheckError();
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("conserved variable update: %5.3f ms\n", elapsedTime);
  #endif    

  #ifdef DE
  Sync_Energies_1D<<<dimGrid,dimBlock>>>(dev_conserved, n_cells, n_ghost, gama);
  #endif


  #ifdef COOLING_GPU
  cooling_kernel<<<dimGrid,dimBlock>>>(dev_conserved, nx, ny, nz, n_ghost, dt, gama, coolTexObj, heatTexObj);
  #endif

  // Calculate the next timestep
  Calc_dt_1D<<<dimGrid,dimBlock>>>(dev_conserved, n_cells, n_ghost, dx, dev_dti_array, gama);
  CudaCheckError();


  // copy the conserved variable array back to the CPU
  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif
  CudaSafeCall( cudaMemcpy(host_conserved, dev_conserved, n_fields*n_cells*sizeof(Real), cudaMemcpyDeviceToHost) );
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("GPU return: %5.3f ms\n", elapsedTime);
  #endif


  #ifdef TIME
  cudaEventRecord(start, 0);
  #endif      
  // copy the dti array onto the CPU
  CudaSafeCall( cudaMemcpy(host_dti_array, dev_dti_array, ngrid*sizeof(Real), cudaMemcpyDeviceToHost) );
  // iterate through to find the maximum inverse dt for this subgrid block
  for (int i=0; i<ngrid; i++) {
    max_dti = fmax(max_dti, host_dti_array[i]);
  }
  #ifdef TIME
  // get stop time and display the timing results
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("dti copying & calc: %5.3f ms\n", elapsedTime);
  #endif     


  // free the CPU memory
  free(host_dti_array);

  #ifdef TEST
  free(test1);
  free(test2);
  #endif

  // free the GPU memory
  cudaFree(dev_conserved);
  cudaFree(Q_L);
  cudaFree(Q_R);
  cudaFree(F);
  cudaFree(etah);
  #ifdef COOLING_GPU
  // Destroy texture object
  cudaDestroyTextureObject(coolTexObj);
  cudaDestroyTextureObject(heatTexObj);
  // Free device memory
  cudaFreeArray(cuCoolArray);
  cudaFreeArray(cuHeatArray);  
  #endif  

  #ifdef TIME
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  #endif


  // return the maximum inverse timestep
  return max_dti;


}












#endif //CUDA
