/*! \file ppmp_vl_cuda.h
 *  \brief Declarations of the cuda van leer ppm kernels. */

#ifdef CUDA

#ifndef PPMP_VL_CUDA_H
#define PPMP_VL_CUDA_H


#include"global.h"

/*! \fn __global__ void PPMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir)
 *  \brief When passed a stencil of conserved variables, returns the left and right 
           boundary values for the interface calculated using ppm. */
__global__ void PPMP_VL(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real gamma, int dir);

/*! \fn __device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
 *  \brief Calculates the limited slope across a cell.*/
__device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo);

/*! \fn __device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R)
 *  \brief Calculates the left and right interface values for a cell using parabolic reconstruction
           in the primitive variables with limited slopes provided. Applies further monotonicity constraints.*/
__device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R);



#endif // PPMP_VL_CUDA_H
#endif // CUDA
