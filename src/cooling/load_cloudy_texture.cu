/*! \file load_cloudy_texture.cu
 *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */

#ifdef CUDA
  #ifdef CLOUDY_COOL

    #include <stdio.h>
    #include <stdlib.h>

    #include "../cooling/cooling_cuda.h"
    #include "../cooling/load_cloudy_texture.h"
    #include "../cooling/texture_utilities.h"
    #include "../global/global.h"
    #include "../global/global_cuda.h"
    #include "../io/io.h"  // provides chprintf

cudaArray *cuCoolArray;
cudaArray *cuHeatArray;

void Test_Cloudy_Textures();
void Test_Cloudy_Speed();

/* \fn void Host_Read_Cooling_Tables(float* cooling_table, float* heating_table)
 * \brief Load the Cloudy cooling tables into host (CPU) memory. */
void Host_Read_Cooling_Tables(float *cooling_table, float *heating_table)
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
  char *pch;

  // allocate arrays for temperature data
  n_arr = (double *)malloc(nx * ny * sizeof(double));
  T_arr = (double *)malloc(nx * ny * sizeof(double));
  L_arr = (double *)malloc(nx * ny * sizeof(double));
  H_arr = (double *)malloc(nx * ny * sizeof(double));

  // Read in cloudy cooling/heating curve (function of density and temperature)
  i = 0;

  const char *cloudy_filename1 = "./cloudy_coolingcurve.txt";
  const char *cloudy_filename2 = "src/cooling/cloudy_coolingcurve.txt";
  const char *file_in_use;

  infile      = fopen(cloudy_filename1, "r");
  file_in_use = cloudy_filename1;
  if (infile == NULL) {
    infile      = fopen(cloudy_filename2, "r");
    file_in_use = cloudy_filename2;
  }

  if (infile == NULL) {
    chprintf(
        "Unable to open Cloudy file with expected relative paths:\n %s \n OR "
        "\n %s\n",
        cloudy_filename1, cloudy_filename2);
    exit(1);
  } else {
    chprintf("Using Cloudy file at relative path: %s \n", file_in_use);
  }

  while (fgets(buffer, sizeof(buffer), infile) != NULL) {
    if (buffer[0] == '#') {
      continue;
    } else {
      pch      = strtok(buffer, "\t");
      n_arr[i] = atof(pch);
      while (pch != NULL) {
        pch = strtok(NULL, "\t");
        if (pch != NULL) T_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL) L_arr[i] = atof(pch);
        pch = strtok(NULL, "\t");
        if (pch != NULL) H_arr[i] = atof(pch);
      }
      i++;
    }
  }
  fclose(infile);

  // copy data from cooling array into the table
  for (i = 0; i < nx * ny; i++) {
    cooling_table[i] = float(L_arr[i]);
    heating_table[i] = float(H_arr[i]);
  }

  // Free arrays used to read in table data
  free(n_arr);
  free(T_arr);
  free(L_arr);
  free(H_arr);
}

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
  GPU_Error_Check(cudaHostAlloc(&cooling_table, nx * ny * sizeof(float), cudaHostAllocDefault));
  GPU_Error_Check(cudaHostAlloc(&heating_table, nx * ny * sizeof(float), cudaHostAllocDefault));

  // Read cooling tables into the host arrays
  Host_Read_Cooling_Tables(cooling_table, heating_table);

  // Allocate CUDA arrays in device memory
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  GPU_Error_Check(cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny));
  GPU_Error_Check(cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny));

  // Copy the cooling and heating arrays from host to device

  // cudaMemcpyToArray is being deprecated
  // cudaMemcpyToArray(cuCoolArray, 0, 0, cooling_table, nx*ny*sizeof(float),
  // cudaMemcpyHostToDevice); cudaMemcpyToArray(cuHeatArray, 0, 0,
  // heating_table, nx*ny*sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy2DToArray(cuCoolArray, 0, 0, cooling_table, nx * sizeof(float), nx * sizeof(float), ny,
                      cudaMemcpyHostToDevice);
  cudaMemcpy2DToArray(cuHeatArray, 0, 0, heating_table, nx * sizeof(float), nx * sizeof(float), ny,
                      cudaMemcpyHostToDevice);

  // Specify textures
  struct cudaResourceDesc coolResDesc;
  memset(&coolResDesc, 0, sizeof(coolResDesc));
  coolResDesc.resType         = cudaResourceTypeArray;
  coolResDesc.res.array.array = cuCoolArray;
  struct cudaResourceDesc heatResDesc;
  memset(&heatResDesc, 0, sizeof(heatResDesc));
  heatResDesc.resType         = cudaResourceTypeArray;
  heatResDesc.res.array.array = cuHeatArray;

  // Specify texture object parameters (same for both tables)
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;  // out-of-bounds fetches return border values
                                                  // dimension 0
  texDesc.addressMode[1] = cudaAddressModeClamp;  // out-of-bounds fetches return border values
                                                  // dimension 1
  texDesc.filterMode = cudaFilterModePoint;
  // We use point mode instead of Linear mode in order to do the interpolation
  // ourselves. Linear mode introduces errors since it only uses 8 bits.
  // cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  // Do not normalize coordinates, in order to simplify conversion from real
  // values to texture coordinates
  texDesc.normalizedCoords = 0;

  // Create texture objects
  cudaCreateTextureObject(&coolTexObj, &coolResDesc, &texDesc, NULL);
  cudaCreateTextureObject(&heatTexObj, &heatResDesc, &texDesc, NULL);

  // Free the memory associated with the cooling tables on the host
  GPU_Error_Check(cudaFreeHost(cooling_table));
  GPU_Error_Check(cudaFreeHost(heating_table));

  // Run Test
  // Test_Cloudy_Textures();
  // Test_Cloudy_Speed();
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

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
__global__ void Test_Cloudy_Textures_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
                                            cudaTextureObject_t heatTexObj)
{
  int id, id_n, id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id / num_n;
  id_n = id % num_n;

  float grid_offset = 0.1 / 512.0;
  // Min value, but include id=-1 as an outside value to check clamping. Use dx
  // = 0.05 instead of 0.1 to check interpolation
  float log_T = 1.0 + (id_T - 1) * 0.05 + grid_offset;
  float log_n = -6.0 + (id_n - 1) * 0.05 + grid_offset;

  // Remap for texture with normalized coords
  // float rlog_T = (log_T - 1.0) / 8.1;
  // float rlog_n = (log_n + 6.0) / 12.1;

  // Remap for texture without normalized coords
  float rlog_T = (log_T - 1.0) * 10;
  float rlog_n = (log_n + 6.0) * 10;

  // Evaluate
  float lambda = Bilinear_Texture(coolTexObj, rlog_T, rlog_n);  // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat   = Bilinear_Texture(heatTexObj, rlog_T, rlog_n);  // tex2D<float>(heatTexObj, rlog_T, rlog_n);

  // Hackfully print it out for processing for correctness
  printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n", log_T, log_n, lambda, heat);
}

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
__global__ void Test_Cloudy_Speed_Kernel(int num_n, int num_T, cudaTextureObject_t coolTexObj,
                                         cudaTextureObject_t heatTexObj)
{
  int id, id_n, id_T;
  id = threadIdx.x + blockIdx.x * blockDim.x;
  // Calculate log_T and log_n based on id
  id_T = id / num_n;
  id_n = id % num_n;

  // Min value, but include id=-1 as an outside value to check clamping. Use dx
  // = 0.05 instead of 0.1 to check interpolation float log_T = 1.0  +
  // (id_T-1)*0.05;
  //  float log_n = -6.0 + (id_n-1)*0.05;

  // Remap for texture with normalized coords
  // float rlog_T = (log_T - 1.0) / 8.1;
  // float rlog_n = (log_n + 6.0) / 12.1;

  // Remap for texture without normalized coords
  // float rlog_T = (log_T - 1.0) * 10;
  // float rlog_n = (log_n + 6.0) * 10;

  float rlog_T = (id_T - 1) * 0.0125;
  float rlog_n = (id_n - 1) * 0.0125;

  // Evaluate
  float lambda = Bilinear_Texture(coolTexObj, rlog_T, rlog_n);  // tex2D<float>(coolTexObj, rlog_T, rlog_n);
  float heat   = Bilinear_Texture(heatTexObj, rlog_T, rlog_n);  // tex2D<float>(heatTexObj, rlog_T, rlog_n);

  // Hackfully print it out for processing for correctness
  // printf("TEST_Cloudy: %.17e %.17e %.17e %.17e \n",log_T, log_n, lambda,
  // heat);
}

/* Consider this function only to be used at the end of Load_Cuda_Textures when
 * testing Evaluate texture on grid of size num_n num_T for variables n,T */
void Test_Cloudy_Textures()
{
  int num_n = 1 + 2 * 121;
  int num_T = 1 + 2 * 81;
  dim3 dim1dGrid((num_n * num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Test_Cloudy_Textures_Kernel, dim1dGrid, dim1dBlock, 0, 0, num_n, num_T, coolTexObj, heatTexObj);
  GPU_Error_Check(cudaDeviceSynchronize());
  printf("Exiting due to Test_Cloudy_Textures() being called \n");
  exit(0);
}

void Test_Cloudy_Speed()
{
  int num_n = 1 + 80 * 121;
  int num_T = 1 + 80 * 81;
  dim3 dim1dGrid((num_n * num_T + TPB - 1) / TPB, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_start = Get_Time();
  for (int i = 0; i < 100; i++) {
    hipLaunchKernelGGL(Test_Cloudy_Speed_Kernel, dim1dGrid, dim1dBlock, 0, 0, num_n, num_T, coolTexObj, heatTexObj);
  }
  GPU_Error_Check(cudaDeviceSynchronize());
  Real time_end = Get_Time();
  printf(" Cloudy Test Time %9.4f micro-s \n", (time_end - time_start));
  printf("Exiting due to Test_Cloudy_Speed() being called \n");
  exit(0);
}

  #endif
#endif
