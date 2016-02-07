/*! \file ppmc_vl_cuda.h
 *  \brief Declarations of the cuda van leer ppmc kernels. */

#ifdef CUDA
#ifdef PPMC

#ifndef PPMC_VL_CUDA_H
#define PPMC_VL_CUDA_H


#include"global.h"

/*! \fn __global__ void PPMC_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm. */
__global__ void PPMC_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir);


#endif // PPMC_VL_CUDA_H
#endif // PPMC
#endif // CUDA
