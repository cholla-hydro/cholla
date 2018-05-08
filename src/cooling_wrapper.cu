/*! \file cooling_wrapper.cu
 *  \brief Wrapper file for to load CUDA cooling tables. */

#ifdef CUDA
#ifdef CLOUDY_COOL

#include<stdio.h>
#include<stdlib.h>
#include"global.h"
#include"cooling_wrapper.h"
#include"cooling_cuda.h"

texture<float, 2, cudaReadModeElementType> coolTexObj;
texture<float, 2, cudaReadModeElementType> heatTexObj;
cudaArray* cuCoolArray;
cudaArray* cuHeatArray;

__global__ void texture_test_kernel();

/* \fn void Load_Cuda_Textures()
 * \brief Load the Cloudy cooling tables into texture memory on the GPU. */
void Load_Cuda_Textures()
{

  float *cooling_table;
  float *heating_table;
  const int nx = 81;
  const int ny = 121;

  // allocate host arrays to be copied to textures
  // these arrays are declared as external pointers in global.h
  cooling_table = (float *) malloc(nx*ny*sizeof(float));
  heating_table = (float *) malloc(nx*ny*sizeof(float));

  // Load cooling tables into the host arrays
  Load_Cooling_Tables(cooling_table, heating_table);

  // Allocate CUDA arrays in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny);
  cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny);
  // Copy to device memory the cooling and heating arrays
  // in host memory
  cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToArray(cuHeatArray, 0, 0, heating_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);


  // Specify texture reference parameters (same for both tables)
  coolTexObj.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  coolTexObj.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  coolTexObj.filterMode = cudaFilterModeLinear; // bi-linear interpolation
  coolTexObj.normalized = true;
  heatTexObj.addressMode[0] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  heatTexObj.addressMode[1] = cudaAddressModeClamp; // out-of-bounds fetches return border values
  heatTexObj.filterMode = cudaFilterModeLinear; // bi-linear interpolation
  heatTexObj.normalized = true;

  cudaBindTextureToArray(coolTexObj, cuCoolArray);
  cudaBindTextureToArray(heatTexObj, cuHeatArray);
/*
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
  texDesc.filterMode = cudaFilterModeLinear; // bi-linear interpolation
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  // Create texture objects
  //cudaTextureObject_t coolTexObj = 0;
  //cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
  //cudaTextureObject_t heatTexObj = 0;
  //cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);
*/
  //texture_test_kernel<<<1,1>>>(); 

  // Free the memory associated with the cooling tables on the host
  free(cooling_table);
  free(heating_table);

}



__global__ void texture_test_kernel()
{
   // Calculate normalized texture coordinates
  float log_n = 1.0;
  float log_T = 6.0;
  log_T = (log_T - 1.0)/8.1;
  log_n = (log_n + 6.0)/12.1;
  printf("n: %e T: %e Texture value = %e\n", log_n, log_T, tex2D<float>(coolTexObj, log_T, log_n));
}

void bind_texture(float* cudaArray) {

}

/* \fn __device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t coolTexObj, cudaTextureObject_t heatTexObj)
 * \brief Uses texture mapping to interpolate Cloudy cooling/heating 
          tables at z = 0 with solar metallicity and an HM05 UV background. */
/*
__device__ Real Cloudy_cool(Real n, Real T)
{
  Real lambda = 0.0; //cooling rate, erg s^-1 cm^3
  Real H = 0.0; //heating rate, erg s^-1 cm^3
  Real cool = 0.0; //cooling per unit volume, erg /s / cm^3
  float log_n, log_T;
  log_n = log10(n);
  log_T = log10(T);
  log_n = 1.0;
  log_T = 6.0;

  // don't allow cooling at super low temps
  if (log_T < 1.0) return cool;

  // remap coordinates for texture
  log_T = (log_T - 1.0)/8.1;
  log_n = (log_n + 6.0)/12.1; 
  printf("n: %f  T: %f  Texture Value: %e\n", log_n, log_T, tex2D<float>(coolTexObj, log_T, log_n));
 
  // don't cool below 10^4 K
  if (log10(T) > 4.0) {
  lambda = tex2D<float>(coolTexObj, log_T, log_n);
  }
  else lambda = 0.0;
  H = tex2D<float>(heatTexObj, log_T, log_n);

  // cooling rate per unit volume
  cool = n*n*(powf(10, lambda) - powf(10, H));

  return cool;
}
*/

/* \fn void Load_Cooling_Tables(float* cooling_table, float* heating_table)
 * \brief Load the Cloudy cooling tables into host (CPU) memory. */
void Load_Cooling_Tables(float* cooling_table, float* heating_table)
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
  infile = fopen("./cloudy_coolingcurve.txt", "r");
  if (infile == NULL) {
    printf("Unable to open Cloudy file.\n");
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


void Free_Cuda_Textures()
{
  // unbind the cuda textures
  cudaUnbindTexture(coolTexObj);
  cudaUnbindTexture(heatTexObj);

  // Free the device memory associated with the cuda arrays
  cudaFreeArray(cuCoolArray);
  cudaFreeArray(cuHeatArray);

}


#endif
#endif
