/*! \file error_check_cuda.cu
 *  \brief Error Check Cuda */

#ifdef CUDA

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>

  #include "../global/global.h"
  #include "../global/global_cuda.h"
  #include "../io/io.h"
  #include "../utils/error_check_cuda.h"
  #include "../utils/gpu.hpp"

__global__ void Check_Value_Along_Axis(Real *dev_array, int n_field, int nx, int ny, int nz, int n_ghost,
                                       int *return_value)
{
  int tid_j = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_k = blockIdx.y * blockDim.y + threadIdx.y;

  if (blockDim.x != N_Y || blockDim.y != N_Z) {
    if (tid_j == 0 && tid_k == 0) {
      printf("ERROR CHECK: Block Dimension Error \n");
    }
    return;
  }

  __shared__ Real sh_data[N_Z * N_Y];
  //
  int n_cells, indx_x, indx_3d, indx_2d;
  Real field_value;

  n_cells = nx * ny * nz;

  int i;
  int error = 0;

  indx_x = 0;
  for (indx_x = 0; indx_x < nx; indx_x++) {
    indx_3d          = indx_x + tid_j * nx + tid_k * nx * ny;
    indx_2d          = tid_j + tid_k * ny;
    field_value      = dev_array[n_field * n_cells + indx_3d];
    sh_data[indx_2d] = field_value;

    __syncthreads();

    if (tid_j == 0 && tid_k == 0) {
      for (i = 0; i < N_Y * N_Z - 1; i++) {
        if (sh_data[i] == sh_data[i + 1]) {
          error += 1;
        }
      }
    }
  }

  if (tid_j == 0 && tid_k == 0) {
    *return_value = error;
  }
}

int Check_Field_Along_Axis(Real *dev_array, int n_field, int nx, int ny, int nz, int n_ghost, dim3 Grid_Error,
                           dim3 Block_Error)
{
  int *error_value_dev;
  CudaSafeCall(cudaMalloc((void **)&error_value_dev, sizeof(int)));
  hipLaunchKernelGGL(Check_Value_Along_Axis, Grid_Error, Block_Error, 0, 0, dev_conserved, 0, nx, ny, nz, n_ghost,
                     error_value_dev);

  int error_value_host;
  CudaSafeCall(cudaMemcpy(&error_value_host, error_value_dev, sizeof(int), cudaMemcpyDeviceToHost));

  return error_value_host;
}

#endif
