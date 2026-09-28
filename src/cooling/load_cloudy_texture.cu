/*! \file load_cloudy_texture.cu
 *  \brief Wrapper file to load cloudy cooling table as CUDA texture. */

#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>

#include "../cooling/cooling_cuda.h"
#include "../cooling/load_cloudy_texture.h"
#include "../cooling/texture_utilities.h"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../io/io.h"  // provides chprintf
#include "../utils/error_handling.h"

/* \fn void Host_Read_Cooling_Tables(float* cooling_table, float* heating_table)
 * \brief Load the Cloudy cooling tables into host (CPU) memory. */
void Host_Read_Cooling_Tables(float *cooling_table, float *heating_table, std::string filename)
{
  int i;
  int nx = 121;
  int ny = 81;

  FILE *infile;
  char buffer[0x1000];
  char *pch;

  // allocate arrays for temperature data
  std::vector<double> n_arr(nx * ny);
  std::vector<double> T_arr(nx * ny);
  std::vector<double> L_arr(nx * ny);
  std::vector<double> H_arr(nx * ny);

  // Read in cloudy cooling/heating curve (function of density and temperature)
  if (not filename.empty()) {
    infile = fopen(filename.c_str(), "r");
    CHOLLA_ASSERT(infile != nullptr, "Unable to open cloudy file at %s", filename.c_str());
  } else {
    const char *cloudy_filename1 = "./cloudy_coolingcurve.txt";
    const char *cloudy_filename2 = "src/cooling/cloudy_coolingcurve.txt";
    const char *file_in_use      = cloudy_filename1;

    infile = fopen(cloudy_filename1, "r");
    if (infile == nullptr) {
      infile      = fopen(cloudy_filename2, "r");
      file_in_use = cloudy_filename2;
    }

    CHOLLA_ASSERT(infile != nullptr,
                  "Unable to open cloudy file. Since no file-path was specified, we tried both \n ->%s AND\n -> %s",
                  cloudy_filename1, cloudy_filename2);
    chprintf("Since no file-path was specified, using Cloudy file at relative path: %s\n", file_in_use);
  }

  i = 0;
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
}

/*! \brief Load the Cloudy cooling tables into texture memory on the GPU.
 *
 *  We'll probably need to factor out some logic to implement metal tables
 */
static void Load_Cuda_Textures_(std::string filename, cudaTextureObject_t &coolTexObj, cudaTextureObject_t &heatTexObj)
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
  Host_Read_Cooling_Tables(cooling_table, heating_table, filename);

  // Allocate CUDA arrays in device memory
  cudaArray *cuCoolArray;
  cudaArray *cuHeatArray;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  GPU_Error_Check(cudaMallocArray(&cuCoolArray, &channelDesc, nx, ny));
  GPU_Error_Check(cudaMallocArray(&cuHeatArray, &channelDesc, nx, ny));

  // note: while we allow cuCoolArray and cuHeatArray to go out of scope in this
  //       function, references are retained to the underlying memory within the cuda
  //       data structures associated with the texture objects we will be initializing.
  //       Thus, when we deallocate the texture objects, we use those references to
  //       deallocate the memory

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
}

static void Free_Single_Cuda_Texture(cudaTextureObject_t &texObj)
{
  // get the handle for the device memory associated with the texture
  cudaResourceDesc resDesc;
  cudaGetTextureObjectResourceDesc(&resDesc, texObj);
  cudaArray *cuArray = resDesc.res.array.array;

  // unbind the cuda textures
  cudaDestroyTextureObject(texObj);

  // Free the device memory associated with the texture
  cudaFreeArray(cuArray);
}

__host__ cool_component::CloudyHeatAndCool::CloudyHeatAndCool(std::string filename, bool enable_heating)
  : enable_heating_(enable_heating)
{
  cudaTextureObject_t coolTexObj = 0;
  cudaTextureObject_t heatTexObj = 0;
  Load_Cuda_Textures_(filename, coolTexObj, heatTexObj);

  // define the deleter callback
  auto deleter = [](cudaTextureObject_t &texObj) { Free_Single_Cuda_Texture(texObj); };

  // actually construct the SharedHandles
  this->coolTexObj_ = SharedHandle<cudaTextureObject_t>(coolTexObj, deleter);
  this->heatTexObj_ = SharedHandle<cudaTextureObject_t>(heatTexObj, deleter);
}
