/*! \file cooling_cuda.h
 *  \brief Declarations of cooling functions. */

#ifdef CUDA
#ifdef COOLING_GPU

#ifndef COOLING_CUDA_H
#define COOLING_CUDA_H

#include "../utils/gpu.hpp"
#include <math.h>
#include "../global/global.h"


/*! \fn void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the value
           of the total energy for each cell according to the specified cooling function. */
void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array);



/*! \fn void Cooling_Calc_dt(Real *dev_dt_array)
 *  \brief Calculate cooling-defined minimum timestep */
Real Cooling_Calc_dt(Real *d_dt_array, Real *h_dt_array, int nx, int ny, int nz);



/*! \fn void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust the value
           of the total energy for each cell according to the specified cooling function. */
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma, Real *dt_array);


/* \fn __device__ Real test_cool(Real n, Real T)
 * \brief Cooling function from Creasey 2011. */
__device__ Real test_cool(int tid, Real n, Real T);


/* \fn __device__ Real primordial_cool(Real n, Real T)
 * \brief Primordial hydrogen/helium cooling curve
          derived according to Katz et al. 1996. */
__device__ Real primordial_cool(Real n, Real T);


/* \fn __device__ Real CIE_cool(Real n, Real T)
 * \brief Analytic fit to a solar metallicity CIE cooling curve
          calculated using Cloudy. */
__device__ Real CIE_cool(Real n, Real T);


/* \fn __device__ Real Cloudy_cool(Real n, Real T)
 * \brief Uses texture mapping to interpolate Cloudy cooling/heating
          tables at z = 0 with solar metallicity and an HM05 UV background. */
__device__ Real Cloudy_cool(Real n, Real T);


#endif //COOLING_CUDA_H
#endif //COOLING_GPU
#endif //CUDA
