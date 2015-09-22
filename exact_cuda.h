/*! \file exact_cuda.h
 *  \brief Declarations of functions for the cuda exact riemann solver kernel. */

#ifdef CUDA

#ifndef EXACT_CUDA_H
#define EXACT_CUDA_H

#include"global.h"


/*! \fn Calculate_Exact_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief Exact Riemann solver based on the Fortran code given in Sec. 4.9 of Toro (1999). */
__global__ void Calculate_Exact_Fluxes(Real *dev_bounds_L, Real *dev_bounds_R, Real *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir);

__device__ Real guessp(Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);

__device__ void prefun(Real *f, Real *fd, Real p, Real dk, Real pk, Real ck, Real gamma);

__device__ void starpv(Real *p, Real *v, Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);

__device__ void sample(const Real pm, const Real vm, Real *d, Real *v, Real *p,
                       Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);


#endif //EXACT_CUDA_H
#endif //CUDA
