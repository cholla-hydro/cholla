/*! \file simple_2D_cuda.h
 *  \brief Declarations for the cuda version of the 2D simple algorithm. */

#ifdef CUDA

#ifndef SIMPLE_2D_CUDA_H
#define SIMPLE_2D_CUDA_H

#include "../global/global.h"

void Simple_Algorithm_2D_CUDA(Real *d_conserved, int nx, int ny, int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound, Real ybound, Real dt, int n_fields);

void Free_Memory_Simple_2D();

#endif //SIMPLE_2D_CUDA_H
#endif //CUDA
