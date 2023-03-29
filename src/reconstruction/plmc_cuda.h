/*! \file plmc_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
 * version. */

#ifndef PLMC_CUDA_H
#define PLMC_CUDA_H

#include "../global/global.h"
#include "../grid/grid_enum.h"

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
__global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir, int n_fields);

namespace plmc_utils
{
struct PlmcPrimitive {
  // Hydro variables
  Real density, velocity_x, velocity_y, velocity_z, pressure;

#ifdef MHD
  Real magnetic_y, magnetic_z;
#endif  // MHD

#ifdef DE
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR
};

PlmcPrimitive __device__ __host__ Load_Data(Real const *dev_conserved, size_t const &xid, size_t const &yid,
                                            size_t const &zid, size_t const &nx, size_t const &ny,
                                            size_t const &n_cells, size_t const &o1, size_t const &o2, size_t const &o3,
                                            Real const &gamma);
}  // namespace plmc_utils
#endif  // PLMC_CUDA_H
