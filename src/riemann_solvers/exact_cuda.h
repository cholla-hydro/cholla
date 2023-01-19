/*! \file exact_cuda.h
 *  \brief Declarations of functions for the cuda exact riemann solver kernel.
 */

#ifdef CUDA

  #ifndef EXACT_CUDA_H
    #define EXACT_CUDA_H

    #include "../global/global.h"

/*! \fn Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L, Real *dev_bounds_R, Real
 * *dev_flux, int nx, int ny, int nz, int n_ghost, Real gamma, int dir, int
 * n_fields) \brief Exact Riemann solver based on the Fortran code given in
 * Sec. 4.9 of Toro (1999). */
__global__ void Calculate_Exact_Fluxes_CUDA(Real *dev_bounds_L,
                                            Real *dev_bounds_R, Real *dev_flux,
                                            int nx, int ny, int nz, int n_ghost,
                                            Real gamma, int dir, int n_fields);

__device__ Real guessp_CUDA(Real dl, Real vxl, Real pl, Real cl, Real dr,
                            Real vxr, Real pr, Real cr, Real gamma);

__device__ void prefun_CUDA(Real *f, Real *fd, Real p, Real dk, Real pk,
                            Real ck, Real gamma);

__device__ void starpv_CUDA(Real *p, Real *v, Real dl, Real vxl, Real pl,
                            Real cl, Real dr, Real vxr, Real pr, Real cr,
                            Real gamma);

__device__ void sample_CUDA(const Real pm, const Real vm, Real *d, Real *v,
                            Real *p, Real dl, Real vxl, Real pl, Real cl,
                            Real dr, Real vxr, Real pr, Real cr, Real gamma);

  #endif  // EXACT_CUDA_H
#endif    // CUDA
