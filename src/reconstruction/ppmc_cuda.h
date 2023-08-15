/*! \file ppmc_cuda.h
 *  \brief Declarations of the cuda ppm kernels, characteristic reconstruction
 * version. */

#ifndef PPMC_CUDA_H
#define PPMC_CUDA_H

#include "../global/global.h"

/*!
 * \brief Computes the left and right interface states using PPM with limiting in the characteristic variables and
 * characteristic tracing. Used for the CTU and SIMPLE integrators. This uses the PPM method described in
 * Stone et al. 2008 "Athena: A New Code for Astrophysical MHD". Fundementally this method relies on a Van Leer limiter
 * in the characteristic variables to monotonize the slopes followed by limiting the interface states using the limiter
 * from Colella & Woodward 1984.
 *
 * \param[in] dev_conserved The conserved variable array
 * \param[out] dev_bounds_L The array of left interfaces
 * \param[out] dev_bounds_R The array of right interfaces
 * \param[in] nx The number of cells in the X-direction
 * \param[in] ny The number of cells in the Y-direction
 * \param[in] nz The number of cells in the Z-direction
 * \param[in] dx The length of the cells in the `dir` direction
 * \param[in] dt The time step
 * \param[in] gamma The adiabatic index
 * \param[in] dir The direction to reconstruct. 0=X, 1=Y, 2=Z
 */
__global__ void PPMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, Real dx,
                         Real dt, Real gamma, int dir);

/*!
 * \brief Computes the left and right interface states using PPM with limiting in the characteristic variables. Used for
 * the VL (Van Leer) integrators. This uses the PPM method described in
 * Felker & Stone 2018 "A fourth-order accurate finite volume method for ideal MHD via upwind constrained transport".
 * This method computes the 3rd order interface then applies a mixture of monoticity constraints from from Colella &
 * Sekora 2008, McCorquodale & Colella 2011, and Colella et al. 2011; for details see the
 * `reconstruction::PPM_Single_Variable` function. We found that this newer method and limiters was more stable, less
 * oscillatory, and faster than the method described in Stone et al. 2008 which is used in PPMC_CTU. The difference is
 * most pronounced in the Brio & Wu shock tube where the PPM oscillations are much smaller using this method.
 *
 * \param[in] dev_conserved The conserved variable array
 * \param[out] dev_bounds_L The array of left interfaces
 * \param[out] dev_bounds_R The array of right interfaces
 * \param[in] nx The number of cells in the X-direction
 * \param[in] ny The number of cells in the Y-direction
 * \param[in] nz The number of cells in the Z-direction
 * \param[in] gamma The adiabatic index
 * \param[in] dir The direction to reconstruct. 0=X, 1=Y, 2=Z
 */
__global__ __launch_bounds__(TPB) void PPMC_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx,
                                               int ny, int nz, Real gamma, int dir);

#endif  // PPMC_CUDA_H
