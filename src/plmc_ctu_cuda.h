/*! \file plmc_ctu_cuda.h
 *  \brief Declarations of the cuda plm kernels, characteristic reconstruction version. */

#ifdef CUDA
#ifdef PLMC

#ifndef PLMC_CTU_CUDA_H
#define PLMC_CTU_CUDA_H

#include"global.h"

/*! \fn __global__ void PLMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMC_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir);


#endif // PLMC_CTU_CUDA_H
#endif // PLMC
#endif // CUDA
