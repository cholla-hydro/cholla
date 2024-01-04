/*! \file plmc_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction
 * version. */

#ifndef PLMC_CUDA_H
#define PLMC_CUDA_H

#include "../global/global.h"
#include "../grid/grid_enum.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

/*! \fn __global__ void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real
 *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real
 gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using plm. */
__global__ __launch_bounds__(TPB) void PLMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                 int ny, int nz, Real dx, Real dt, Real gamma, int dir, int n_fields);

#endif  // PLMC_CUDA_H
