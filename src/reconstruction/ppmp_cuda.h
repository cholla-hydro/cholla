/*! \file ppmp_cuda.h
 *  \brief Declarations of the cuda ppmp kernels. */

#ifdef CUDA

#ifndef PPMP_CUDA_H
#define PPMP_CUDA_H


#include "../global/global.h"

/*! \fn __global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir, int n_fields)
 *  \brief When passed a stencil of conserved variables, returns the left and right
           boundary values for the interface calculated using ppm with limiting in the primative variables. */
__global__ void PPMP_cuda(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir, int n_fields);

/*! \fn __device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo)
 *  \brief Calculates the limited slope across a cell.*/
__device__ Real Calculate_Slope(Real q_imo, Real q_i, Real q_ipo);

/*! \fn __device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real *q_L, Real *q_R)
 *  \brief Calculates the left and right interface values for a cell using parabolic reconstruction
           in the primitive variables with limited slopes provided. Applies further monotonicity constraints.*/
__device__ void Interface_Values_PPM(Real q_imo, Real q_i, Real q_ipo, Real del_q_imo, Real del_q_i, Real del_q_ipo, Real *q_L, Real *q_R);

/*! \fn calc_d2_rho
 *  \brief Returns the second derivative of rho across zone i. (Fryxell Eqn 35) */
__device__ Real calc_d2_rho(Real rho_imo, Real rho_i, Real rho_ipo, Real dx);

/*! \fn calc_eta
 *  \brief Returns a dimensionless quantity relating the 1st and 3rd derivatives
    See Fryxell Eqn 36. */
__device__ Real calc_eta(Real d2rho_imo, Real d2rho_ipo, Real dx, Real rho_imo, Real rho_ipo);

#endif // PPMP_CUDA_H
#endif // CUDA
