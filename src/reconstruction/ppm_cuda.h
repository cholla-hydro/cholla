/*! \file ppm_cuda.h
 *  \brief Declarations of the cuda ppm kernels, characteristic reconstruction
 * version. */

#ifndef PPM_CUDA_H
#define PPM_CUDA_H

#include "../global/global.h"

/*!
 * \brief Computes the left and right interface states using PPM with limiting in the primitive or characteristic
 * variables. This uses the PPM method described in Felker & Stone 2018 "A fourth-order accurate finite volume method
 * for ideal MHD via upwind constrained transport". This method computes the 3rd order interface then applies a mixture
 * of monoticity constraints from from Colella & Sekora 2008, McCorquodale & Colella 2011, and Colella et al. 2011. We
 * found that this newer method and limiters was more stable, less oscillatory, and faster than the method described in
 * Stone et al. 2008 which was previously used. The difference is most pronounced in the Brio & Wu shock tube where the
 * PPM oscillations are much smaller using this method.
 *
 * \tparam dir The direction to reconstruct. 0=X, 1=Y, 2=Z
 * \param[in] dev_conserved The conserved variable array
 * \param[out] dev_bounds_L The array of left interfaces
 * \param[out] dev_bounds_R The array of right interfaces
 * \param[in] nx The number of cells in the X-direction
 * \param[in] ny The number of cells in the Y-direction
 * \param[in] nz The number of cells in the Z-direction
 * \param[in] gamma The adiabatic index
 */
template <int dir>
__global__ __launch_bounds__(TPB) void PPM_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                                int ny, int nz, Real dx, Real dt, Real gamma);

#endif  // PPM_CUDA_H
