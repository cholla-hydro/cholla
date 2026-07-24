/*! \file RT_functors.h
 *  \brief Declarations for the gpu RT functors. */

#ifndef RT_FUNCTORS_H
#define RT_FUNCTORS_H

#include "../global/global.h"
#include "../utils/gpu.hpp"
#include "alt/decl.h"
#include "radiation.h"

/*! \brief  Compute pressure tensor
 *    This has as to be a separate kernel since
 *    pij is needed in its entirety for the step */
template <class PijFunctor>
void __global__ GLFMakeP_Kernel(int nx, int ny, int nz, int n_ghost, float dx, const Real* rfi, Real* pij,
                                PijFunctor pf, int deb)
{

  // Noting that here nx, ny, and nz contain 2*n_ghost cells plus the local grid dimensions
  // the input rfi field contains the intensity, Mx, My, and Mz fields for one frequency
  // The output pij contains the pressure fields

  // The fields should be ordered as:
  // *intensity, *intensity_Mx, *intensity_My, and *intensity_Mz

  // Consider re-writing to follow normal GPU thread indexing throughout Cholla

  /*const int nw3 = nx*ny*nz;
  const int tid = threadIdx.x + blockIdx.x*blockDim.x;
  const int nc = nx - 2*n_ghost;
  const int jkc = tid/nc;
  const int ic = n_ghost + tid%nc;
  const int jc = n_ghost + jkc%nc;
  const int kc = n_ghost + jkc/nc;
  if(kc >= nx-n_ghost) return;*/

  // try calc_abs
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int jk  = tid / nx;
  const int i   = tid % nx;
  const int j   = jk % ny;
  const int k   = jk / ny;

  // This limits to real cells only
  if (i < n_ghost || j < n_ghost || k < n_ghost || i >= nx - n_ghost || j >= ny - n_ghost || k >= nz - n_ghost) return;

  const int ic = i;
  const int jc = j;
  const int kc = k;

  pf(n_ghost, nx, ny, nz, ic, jc, kc, rfi, pij, deb);
}
#endif
