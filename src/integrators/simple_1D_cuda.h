/*! \file simple_1D_cuda.h
 *  \brief Declarations for the 1D simple algorithm. */

#ifndef SIMPLE_1D_CUDA_H
#define SIMPLE_1D_CUDA_H

#include "../global/global.h"

void Simple_Algorithm_1D_CUDA(Real *d_conserved, int nx, int x_off, int n_ghost, Real dx, Real xbound, Real dt,
                              int n_fields, int custom_grav);

void Free_Memory_Simple_1D();

#endif  // Simple_1D_CUDA_H
