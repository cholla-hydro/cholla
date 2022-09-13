/*! \file cooling_wrapper.cu
 *  \brief Wrapper file for to load CUDA cooling tables. */

#ifdef CUDA
#ifdef CLOUDY_COOL

#include <stdio.h>
#include <stdlib.h>
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../cooling/cooling_wrapper.h"
#include "../cooling/cooling_cuda.h"

cudaTextureObject_t coolTexObj = 0;
cudaTextureObject_t heatTexObj = 0;
cudaArray* cuCoolArray;
cudaArray* cuHeatArray;

__device__ float lerp(float v0, float v1, float f)
{
  return fma(f, v1, fma(-f,v0,v0));
}
__device__ float Bilinear_Custom(cudaTextureObject_t tex, float x, float y)
{
  float px = floorf(x);
  float py = floorf(y);
  float fx = x - px;
  float fy = y - py;
  px += 0.5;
  py += 0.5;
  float t00 = tex2D<float>(tex,px,py);
  float t01 = tex2D<float>(tex,px,py+1);
  float t10 = tex2D<float>(tex,px+1,py);
  float t11 = tex2D<float>(tex,px+1,py+1);
  return lerp(lerp(t00, t10, fx), lerp(t01, t11, fx), fy);
   
}
/* Consider this function only to be used at the end of Load_Cuda_Textures when testing
 * Evaluate texture on grid of size num_n num_T for variables n,T */
__global__ void Test_Cloudy_Textures_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
{
  int id,id_n,id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id/num_n;
  id_n = id%num_n;

  // Min value, but include id=-1 as an outside value to check clamping. Use dx = 0.05 instead of 0.1 to check interpolation
  float log_T = 1.0  + (id_T-1)*0.05;
  float log_n = -6.0 + (id_n-1)*0.05;
    
  // Remap for texture with normalized coords
  // float rlog_T = (log_T - 1.0) / 8.1;
  // float rlog_n = (log_n + 6.0) / 12.1;

  // Remap for texture without normalized coords
  float rlog_T = (log_T - 1.0) * 10;
  float rlog_n = (log_n + 6.0) * 10;  
  
  // Evaluate
  float lambda = Bilinear_Custom(coolTexObj, rlog_T, rlog_n); // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat = Bilinear_Custom(heatTexObj, rlog_T, rlog_n); // tex2D<float>(heatTexObj, rlog_T, rlog_n);  

  // Hackfully print it out for processing for correctness
  printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n",log_T, log_n, lambda, heat);
    
}

/* Consider this function only to be used at the end of Load_Cuda_Textures when testing
 * Evaluate texture on grid of size num_n num_T for variables n,T */
void Test_Cloudy_Textures()
{
  int num_n = 1+2*121;
  int num_T = 1+2*81;
  dim3 dim1dGrid((num_n*num_T+TPB-1)/TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Test_Cloudy_Textures_Kernel,dim1dGrid,dim1dBlock,0,0,num_n,num_T,coolTexObj,heatTexObj);
  CHECK(cudaDeviceSynchronize());
  printf("Exiting due to Test_Cloudy_Textures() being called \n");
  exit(0);
}

/* \fn void Host_Read_Cooling_Tables(float* cooling_table, float* heating_table)
 * \brief Load the Cloudy cooling tables into host (CPU) memory. */
void Host_Read_Cooling_Tables(float* cooling_table, float* heating_table)
{
  double *n_arr;
  double *T_arr;
  double *L_arr;
  double *H_arr;

  int i;
  int nx = 121;
  int ny = 81;

  FILE *infile;
  char buffer[0x1000];
  char * pch;

  // allocate arrays for temperature data
  n_arr = (double *) malloc(nx*ny*sizeof(double));
  T_arr = (double *) malloc(nx*ny*sizeof(double));
  L_arr = (double *) malloc(nx*ny*sizeof(double));
  H_arr = (double *) malloc(nx*ny*sizeof(double));

  // Read in cloudy cooling/heating curve (function of density and temperature)
  i=0;
  const char* cloudy_filename = "src/cooling/cloudy_coolingcurve.txt";
  infile = fopen(cloudy_filename, "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file with expected path: %s\n", cloudy_filename);
    exit(1);
  }
  while (fgets(buffer, sizeof(buffer), infile) != NULL)
  {
    if (buffer[0] == '#') {
      continue;
    }
    else {
      pch = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL)
      {
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL)
          H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  // copy data from cooling array into the table
  for (i=0; i<nx*ny; i++)
  {
    cooling_table[i] = float(L_arr[i]);
    heating_table[i] = float(H_arr[i]);
  }

  // Free arrays used to read in table data
  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);
}


/* \fn void Load_Cuda_Textures2()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures()
{
  printf("Initializing Cloudy Textures\n");
  float *cooling_table;
  float *heating_table;
  const int nx = 81;
  const int ny = 121;

  // allocate host arrays to be copied to textures
  // these arrays are declared as external pointers in global.h
  CudaSafeCall( cudaHostAlloc(&cooling_table, nx*ny*sizeof(float), cudaHostAllocDefault) );
  CudaSafeCall( cudaHostAlloc(&heating_table, nx*ny*sizeof(float), cudaHostAllocDefault) );

  // Read cooling tables into the host arrays
  Host_Read_Cooling_Tables(cooling_table, heating_table);

  // Allocate CUDA arrays in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny);
  cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny);

  // Copy the cooling and heating arrays from host to device

  // cudaMemcpyToArray is being deprecated 
  // cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
  // cudaMemcpyToArray(cuHeatArray, 0, 0, heating_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy2DToArray(cuCoolArray, 0, 0, cooling_table, nx*sizeof(float) , nx*sizeof(float), ny, cudaMemcpyHostToDevice);
  cudaMemcpy2DToArray(cuHeatArray, 0, 0, heating_table, nx*sizeof(float) , nx*sizeof(float), ny, cudaMemcpyHostToDevice);
  
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
  texDesc.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values dimension 0
  texDesc.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values dimension 1
  texDesc.filterMode = cudaFilterModePoint;//cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;

  // Create texture objects
  cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
  cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);

  // Free the memory associated with the cooling tables on the host
  CudaSafeCall( cudaFreeHost(cooling_table) );
  CudaSafeCall( cudaFreeHost(heating_table) );

  // Run Test
  Test_Cloudy_Textures();
  
}

void Free_Cuda_Textures()
{
  // unbind the cuda textures
  cudaDestroyTextureObject(coolTexObj);
  cudaDestroyTextureObject(heatTexObj);

  // Free the device memory associated with the cuda arrays
  cudaFreeArray(cuCoolArray);
  cudaFreeArray(cuHeatArray);

}


#endif
#endif
