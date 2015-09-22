/*! \file VL_3D_cuda.h
 *  \brief Declarations for the cuda version of the 3D VL algorithm. */

#ifdef CUDA

#ifndef VL_3D_CUDA_H
#define VL_3D_CUDA_H

#include"global.h"

Real VL_Algorithm_3D_CUDA(Real *host_conserved, int nx, int ny, int nz, int n_ghost, Real dx, Real dy, Real dz, Real dt);


#endif //VL_3D_CUDA_H
#endif //CUDA
