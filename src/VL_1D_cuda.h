/*! \file VL_1D_cuda.h
 *  \brief Declarations for the cuda version of the 1D VL algorithm. */

#ifdef CUDA

#ifndef VL_1D_CUDA_H
#define VL_1D_CUDA_H

#include"global.h"

Real VL_Algorithm_1D_CUDA(Real *host_conserved, int nx, int n_ghost, Real dx, Real dt);


#endif //VL_1D_CUDA_H
#endif //CUDA
