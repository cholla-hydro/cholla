/*! \file hllc_cuda.h
 *  \brief Declarations of functions for the cuda hllc riemann solver kernel. */

#ifdef CUDA

#ifndef HLLC_CUDA_H
#define HLLC_CUDA_H

#include"global.h"


/*! \fn Calculate_HLLC_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief Roe Riemann solver based on the version described in Stone et al, 2008. */
__global__ void Calculate_HLLC_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, Real *dev_etah, int dir);



#endif //HLLC_CUDA_H
#endif //CUDA
