/*! \file RT_functors.h
 *  \brief Declarations for the gpu RT functors. */

//#ifdef CUDA
#ifdef RT

    #ifndef RT_FUNCTORS_H
      #define RT_FUNCTORS_H

      #include "../global/global.h"
      #include "alt/decl.h"
      #include "radiation.h"
      #include "../utils/gpu.hpp"


//  Compute pressure tensor - has to be a separate kernel since
//  pij is needed in its entirety for the step
template<class PijFunctor> void __global__ GLFMakeP_Kernel(int nx, int ny, int nz, int n_ghost, float dx,
                                                    const Real* rfi, Real* pij, PijFunctor pf, int deb)
{
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

    if (i < n_ghost || j < n_ghost || k < n_ghost || i >= nx - n_ghost || j >= ny - n_ghost || k >= nz - n_ghost) return;

    const int ic = i;
    const int jc = j;
    const int kc = k;

    pf(n_ghost,nx,ny,nz,ic,jc,kc,rfi,pij,deb);
}
    #endif 
#endif    // RT
//#endif      // CUDA
