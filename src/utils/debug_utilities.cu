#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../io/io.h"                 // provides chprintf
#include "../utils/error_handling.h"  // provides chexit

__global__ void Check_For_Extreme_Temperature_Kernel(Real* dev_conserved, int n_cells, Real gamma, Real lower_limit, Real upper_limit, int marker, bool* out_bool)
{
  // Calculate temperature
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= n_cells) {
    return;
  }
  const Real d = dev_conserved[id];
  const Real E = dev_conserved[id + n_cells * grid_enum::Energy];
  const Real px = dev_conserved[id + n_cells * grid_enum::momentum_x];
  const Real py = dev_conserved[id + n_cells * grid_enum::momentum_y];
  const Real pz = dev_conserved[id + n_cells * grid_enum::momentum_z];

  const Real E_kinetic = 0.5 * (px * px + py * py + pz * pz) / d;
  /*
  const Real vx = dev_conserved[1 * n_cells + id] / d;
  const Real vy = dev_conserved[2 * n_cells + id] / d;
  const Real vz = dev_conserved[3 * n_cells + id] / d;
  const Real E_kinetic = 0.5 * d * (vx * vx + vy * vy + vz * vz);
  */
  

  const Real E_thermal = E - E_kinetic;
  const Real p = (E_thermal) * (gamma - 1.0);

  const Real mu = 0.6;
  const Real n = d * DENSITY_UNIT / (mu * MP);
  const Real T_kelvin = p * PRESSURE_UNIT / (n * KB);


  if (T_kelvin <= lower_limit || T_kelvin >= upper_limit) {
    out_bool[0] = true;
    kernel_printf("Check_For_Extreme_Temperature_Kernel found Value: %g E: %g E_thermal: %g E_kinetic: %g Marker: %d Thread: %d\n", T_kelvin, E, E_thermal, E_kinetic, marker, id);
  }
}


void Check_For_Extreme_Temperature(Real* dev_conserved, int n_cells, Real gamma, Real lower_limit, Real upper_limit, int marker)
{
  bool host_out_bool[1] = {false};
  bool* out_bool;
  cudaMalloc((void**)&out_bool, sizeof(bool));
  cudaMemcpy(out_bool, host_out_bool, sizeof(bool), cudaMemcpyHostToDevice);
  int ngrid = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  hipLaunchKernelGGL(Check_For_Extreme_Temperature_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, n_cells, gamma, lower_limit, upper_limit, marker, out_bool);

  cudaMemcpy(host_out_bool, out_bool, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaFree(out_bool);

  if (host_out_bool[0]) {
    chexit(-1);
  }
}





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
