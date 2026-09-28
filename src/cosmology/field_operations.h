/*! \file
 *! \brief declares field-wide operations
 */

#pragma once
#include "../global/global.h"
/*! \brief Multiply one field a multiplicative constant */
__global__ void Rescale_Field_GPU(Real *d_x, const Real A, int nx, int ny, int nz, int n_ghost);

/*! \brief Multiply one field elementwise by another */
__device__ void Field_Elementwise_Product_GPU(Real *d_x, Real *d_y, int nx, int ny, int nz, int n_ghost);
// inline __device__ void FFT_Populate_Wavevectors_GPU(Real *d_kx, Real *d_ky, Real *d_kz, Real *d_kk, int n_cells, int
// n_ghost);
