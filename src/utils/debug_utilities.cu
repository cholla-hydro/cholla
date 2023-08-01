#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../io/io.h"                 // provides chprintf
#include "../utils/error_handling.h"  // provides chexit

__global__ void Dump_Values_Kernel(Real* device_array, int array_size, int marker)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= array_size) {
    return;
  }
  kernel_printf("Dump Values: marker %d tid %d value %g \n", marker, tid, device_array[tid]);
}

/*
  Prints out all values of a device_array
 */
void Dump_Values(Real* device_array, int array_size, int marker)
{
  int ngrid = (array_size + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Dump_Values_Kernel, dim1dGrid, dim1dBlock, 0, 0, device_array, array_size, marker);
}

__global__ void Check_For_Nan_Kernel(Real* device_array, int array_size, int check_num, bool* out_bool)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= array_size) {
    return;
  }
  if (device_array[tid] == device_array[tid]) {
    return;
  }
  out_bool[0] = true;
  kernel_printf("Check_For_Nan_Kernel found Nan Checknum: %d Thread: %d\n", check_num, tid);
}

/*
  Checks a device_array for NaN and prints/exits if found
 */
void Check_For_Nan(Real* device_array, int array_size, int check_num)
{
  bool host_out_bool[1] = {false};
  bool* out_bool;
  cudaMalloc((void**)&out_bool, sizeof(bool));
  cudaMemcpy(out_bool, host_out_bool, sizeof(bool), cudaMemcpyHostToDevice);
  int ngrid = (array_size + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  hipLaunchKernelGGL(Check_For_Nan_Kernel, dim1dGrid, dim1dBlock, 0, 0, device_array, array_size, check_num, out_bool);
  cudaMemcpy(host_out_bool, out_bool, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(out_bool);

  if (host_out_bool[0]) {
    chexit(-1);
  }
}
