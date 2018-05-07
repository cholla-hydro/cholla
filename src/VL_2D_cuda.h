/*! \file VL_2D_cuda.h
 *  \brief Declarations for the cuda version of the 2D VL algorithm. */

#ifdef CUDA

#ifndef VL_2D_CUDA_H
#define VL_2D_CUDA_H

#include"global.h"

Real VL_Algorithm_2D_CUDA(Real *host_conserved0, Real *host_conserved1, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, int n_fields);


#endif //VL_2D_CUDA_H
#endif //CUDA
