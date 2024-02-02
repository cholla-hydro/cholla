/*! \file hllc_cuda.h
 *  \brief Declarations of functions for the cuda hllc riemann solver kernel. */

#ifndef HLL_CUDA_H
#define HLL_CUDA_H

#include "../global/global.h"

/*! \fn Calculate_HLLC_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
 * *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir, int
 * n_fields) \brief Roe Riemann solver based on the version described in Stone
 * et al, 2008. */
__global__ void Calculate_HLL_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
                                          int nz, int n_ghost, Real gamma, int dir, int n_fields);

#endif  // HLLC_CUDA_H
