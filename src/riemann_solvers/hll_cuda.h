/*! \file hllc_cuda.h
 *  \brief Declarations of functions for the cuda hllc riemann solver kernel. */

#ifndef HLL_CUDA_H
#define HLL_CUDA_H

#include "../global/global.h"

/*!
 * \brief HLL Riemann solver
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
__global__ void Calculate_HLL_Fluxes_CUDA(Real const *dev_conserved, Real const *dev_bounds_L, Real const *dev_bounds_R,
                                          Real *dev_flux, int const nx, int const ny, int const nz, int const n_cells,
                                          Real const gamma, Real const dx, Real const dt, int const n_fields);

#endif  // HLLC_CUDA_H
