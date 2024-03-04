/*! \file plmc_cuda.cu
 *  \brief Definitions of the piecewise linear reconstruction functions with
           limiting applied in the characteristic variables, as described
           in Stone et al., 2008. */

#include <math.h>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../reconstruction/plmc_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"

#ifdef DE  // PRESSURE_DE
  #include "../utils/hydro_utilities.h"
#endif  // DE

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
template <int dir>
__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                 int ny, int nz, Real dx, Real dt, Real gamma)
{
  // get a thread ID
  int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(thread_id, nx, ny, xid, yid, zid);

  // Ensure that we are only operating on cells that will be used
  if (reconstruction::Thread_Guard<2>(nx, ny, nz, xid, yid, zid)) {
    return;
  }

  auto [interface_L_iph, interface_R_imh] =
      reconstruction::PLMC_Reconstruction<dir>(dev_conserved, xid, yid, zid, nx, ny, nz, dx, dt, gamma);

  // apply minimum constraints
  interface_R_imh.density  = fmax(interface_R_imh.density, (Real)TINY_NUMBER);
  interface_L_iph.density  = fmax(interface_L_iph.density, (Real)TINY_NUMBER);
  interface_R_imh.pressure = fmax(interface_R_imh.pressure, (Real)TINY_NUMBER);
  interface_L_iph.pressure = fmax(interface_L_iph.pressure, (Real)TINY_NUMBER);

  // Set the field indices for the various directions
  int o1, o2, o3;
  if constexpr (dir == 0) {
    o1 = grid_enum::momentum_x;
    o2 = grid_enum::momentum_y;
    o3 = grid_enum::momentum_z;
  } else if constexpr (dir == 1) {
    o1 = grid_enum::momentum_y;
    o2 = grid_enum::momentum_z;
    o3 = grid_enum::momentum_x;
  } else if constexpr (dir == 2) {
    o1 = grid_enum::momentum_z;
    o2 = grid_enum::momentum_x;
    o3 = grid_enum::momentum_y;
  }

  // Compute the total number of cells
  int const n_cells = nx * ny * nz;

  // Convert the left and right states in the primitive to the conserved variables send final values back from kernel
  // bounds_R refers to the right side of the i-1/2 interface
  size_t id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);
  reconstruction::Write_Data(interface_L_iph, dev_bounds_L, dev_conserved, id, n_cells, o1, o2, o3, gamma);

  id = cuda_utilities::compute1DIndex(xid - int(dir == 0), yid - int(dir == 1), zid - int(dir == 2), nx, ny);
  reconstruction::Write_Data(interface_R_imh, dev_bounds_R, dev_conserved, id, n_cells, o1, o2, o3, gamma);
}

// Instantiate the relevant template specifications
template __global__ __launch_bounds__(TPB) void PLMC_cuda<0>(Real *dev_conserved, Real *dev_bounds_L,
                                                             Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                                                             Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PLMC_cuda<1>(Real *dev_conserved, Real *dev_bounds_L,
                                                             Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                                                             Real dt, Real gamma);
template __global__ __launch_bounds__(TPB) void PLMC_cuda<2>(Real *dev_conserved, Real *dev_bounds_L,
                                                             Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                                                             Real dt, Real gamma);
