/*! \file VL_1D_cuda.h
 *  \brief Declarations for the cuda version of the 1D VL algorithm. */

#ifndef VL_1D_CUDA_H
#define VL_1D_CUDA_H

#include "../global/global.h"

void VL_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
                          int n_fields, int custom_grav);

void Free_Memory_VL_1D();

#endif  // VL_1D_CUDA_H
