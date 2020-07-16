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
  CudaSafeCall( cudaHostAlloc(&cooling_table, nx*ny*sizeof(float), cudaHostAllocDefault) );
  CudaSafeCall( cudaHostAlloc(&heating_table, nx*ny*sizeof(float), cudaHostAllocDefault) );

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

  // Free the memory associated with the cooling tables on the host
  CudaSafeCall( cudaFreeHost(cooling_table) );
  CudaSafeCall( cudaFreeHost(heating_table) );

}


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
  infile = fopen("../cooling/cloudy_coolingcurve.txt", "r");
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
