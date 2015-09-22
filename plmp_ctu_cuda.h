/*! \file plmp_ctu_cuda.h
 *  \brief Declarations of the cuda plm kernels, primative variable reconstruction version. */

#ifdef CUDA
#ifdef PLMP

#ifndef PLMP_CTU_CUDA_H
#define PLMP_CTU_CUDA_H


#include"global.h"

/*! \fn __global__ void PLMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir);


#endif // PLMP_CTU_CUDA_H
#endif // PLMP
#endif // CUDA
