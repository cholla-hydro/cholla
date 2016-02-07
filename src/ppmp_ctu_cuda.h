/*! \file ppmp_ctu_cuda.h
 *  \brief Declarations of the cuda ppm kernels. */

#ifdef CUDA
#ifdef PPMP

#ifndef PPMP_CTU_CUDA_H
#define PPMP_CTU_CUDA_H

#include"global.h"


/*! \fn void PPMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir)
 *  \brief Use the piecewise parabolic method to calculate boundary values for each cell. */
__global__ void PPMP_CTU(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int nx, int ny, int nz, int n_ghost, Real dx, Real dt, Real gamma, int dir);


/*! \fn interface_value 
 *  \brief Returns the interpolated value at i | ipo cell interface.*/
__device__ Real interface_value(Real q_imo, Real q_i, Real q_ipo, Real q_ipt, Real dx);


/*! \fn calc_delta_q
 *  \brief Returns the average slope in zone i of the parabola with zone averages
     of imo, i, and ipo. See Fryxell Eqn 24. */
__device__ Real calc_delta_q(Real q_imo, Real q_i, Real q_ipo, Real dx);


/*! \fn limit_delta_q
 *  \brief Limits the value of delta_rho according to Fryxell Eqn 26
     to ensure monotonic interface values. */
__device__ Real limit_delta_q(Real del_in, Real q_imo, Real q_i, Real q_ipo);


/*! \fn calc_d2_rho
 *  \brief Returns the second derivative of rho across zone i. (Fryxell Eqn 35) */
__device__ Real calc_d2_rho(Real rho_imo, Real rho_i, Real rho_ipo, Real dx);


/*! \fn calc_eta
 *  \brief Returns a dimensionless quantity relating the 1st and 3rd derivatives
    See Fryxell Eqn 36. */
__device__ Real calc_eta(Real d2rho_imo, Real d2rho_ipo, Real dx, Real rho_imo, Real rho_ipo);



#endif // PPMP_CTU_CUDA_H
#endif // PPMP
#endif // CUDA
