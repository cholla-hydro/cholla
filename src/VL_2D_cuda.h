/*! \file VL_2D_cuda.h
 *  \brief Declarations for the cuda version of the 2D VL algorithm. */

#ifdef CUDA

#ifndef VL_2D_CUDA_H
#define VL_2D_CUDA_H

#include"global.h"

Real VL_Algorithm_2D_CUDA(Real *host_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real dt);


#endif //VL_2D_CUDA_H
#endif //CUDA
