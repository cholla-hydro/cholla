/*! \file roe_cuda.h
 *  \brief Declarations of functions for the cuda roe riemann solver kernel. */

#ifndef ROE_CUDA_H
#define ROE_CUDA_H

#include "../global/global.h"

/*! \fn Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
 * *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah,
 * int dir, int n_fields) \brief Roe Riemann solver based on the version
 * described in Stone et al, 2008. */
__global__ void Calculate_Roe_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny,
                                          int nz, int n_ghost, Real gamma, int dir, int n_fields);

#endif  // ROE_CUDA_H
