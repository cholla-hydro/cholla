/*! \file exact_cuda.h
 *  \brief Declarations of functions for the cuda exact riemann solver kernel.
 */

#ifndef EXACT_CUDA_H
#define EXACT_CUDA_H

#include "../global/global.h"

/*!
 * \brief Exact Riemann solver based on the Fortran code given in Sec. 4.9 of Toro (1999).
 *
 * \tparam reconstruction What kind of reconstruction to use, PCM, PLMC, etc. This argument should always be a
 * member of the reconstruction::Kind enum, behaviour is undefined otherwise.
 * \tparam direction The direction that the solve is taking place in. 0=X, 1=Y, 2=Z
 * \param[in]  dev_bounds_L The interface states on the left side of the
 * interface
 * \param[in]  dev_bounds_R The interface states on the right side of
 * the interface
 * \param[out] dev_flux The output flux
 * \param[in]  n_cells Total number of cells
 * \param[in]  n_ghost Number of ghost cells on each side
 * \param[in]  n_fields The total number of fields
 */
template <int reconstruction, uint direction>
__global__ void Calculate_Exact_Fluxes_CUDA(Real const *dev_conserved, Real const *dev_bounds_L,
                                            Real const *dev_bounds_R, Real *dev_flux, int const nx, int const ny,
                                            int const nz, int const n_cells, Real const gamma, int const n_fields);

__device__ Real guessp_CUDA(Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);

__device__ void prefun_CUDA(Real *f, Real *fd, Real p, Real dk, Real pk, Real ck, Real gamma);

__device__ void starpv_CUDA(Real *p, Real *v, Real dl, Real vxl, Real pl, Real cl, Real dr, Real vxr, Real pr, Real cr,
                            Real gamma);

__device__ void sample_CUDA(const Real pm, const Real vm, Real *d, Real *v, Real *p, Real dl, Real vxl, Real pl,
                            Real cl, Real dr, Real vxr, Real pr, Real cr, Real gamma);

#endif  // EXACT_CUDA_H
