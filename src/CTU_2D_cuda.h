/*! \file CTU_2D_cuda.h
 *  \brief Declarations for the cuda version of the 2D CTU algorithm. */

#ifdef CUDA

#ifndef CTU_2D_CUDA_H
#define CTU_2D_CUDA_H

#include"global.h"

Real CTU_Algorithm_2D_CUDA(Real *host_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt);


#endif //CTU_2D_CUDA_H
#endif //CUDA
