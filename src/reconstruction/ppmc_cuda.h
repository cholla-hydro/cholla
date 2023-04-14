/*! \file ppmc_cuda.h
 *  \brief Declarations of the cuda ppm kernels, characteristic reconstruction
 * version. */

#ifndef PPMC_CUDA_H
#define PPMC_CUDA_H

#include "../global/global.h"

/*! \fn void PPMC(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R,
 int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and
 right boundary values for the interface calculated using ppm. */
__global__ void PPMC_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                          Real dt, Real gamma, int dir);

#endif  // PPMC_CUDA_H
