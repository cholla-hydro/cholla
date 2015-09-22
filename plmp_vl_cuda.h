/*! \file plmp_vl_cuda.h
 *  \brief Declarations of the cuda van leer plm kernels. */

#ifdef CUDA

#ifndef PLMP_VL_CUDA_H
#define PLMP_VL_CUDA_H


#include"global.h"

/*! \fn __global__ void PLMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using plm. */
__global__ void PLMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir);


/*! \fn __device__ void Interface_Values_PLM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R)
 *  \brief Calculates the left and right interface values for a cell using linear reconstruction
           in the primitive variables with minmod slope limiting. */
__device__ void Interface_Values_PLM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R);


#endif // PLMP_VL_CUDA_H
#endif // CUDA
