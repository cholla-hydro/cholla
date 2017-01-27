/*! \file CTU_1D_cuda.h
 *  \brief Declarations for the cuda version of the 1D CTU algorithm. */

#ifdef CUDA

#ifndef CTU_1D_CUDA_H
#define CTU_1D_CUDA_H

#include"global.h"

Real CTU_Algorithm_1D_CUDA(Real *host_conserved, int nx, int x_off, int n_ghost, Real dx, Real dt);


#endif //CTU_1D_CUDA_H
#endif //CUDA
