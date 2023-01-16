/*! \file cooling_cuda.h
 *  \brief Declarations of cooling functions. */

#ifdef CUDA
  #ifdef COOLING_GPU
    #pragma once

    #include <math.h>

    #include "../global/global.h"
    #include "../utils/gpu.hpp"

extern cudaTextureObject_t coolTexObj;
extern cudaTextureObject_t heatTexObj;

/*! \fn void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int
 n_ghost, int n_fields, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust
 the value of the total energy for each cell according to the specified cooling
 function. */
void Cooling_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost,
                    int n_fields, Real dt, Real gamma);

/*! \fn void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz, int
 n_ghost, Real dt, Real gamma)
 *  \brief When passed an array of conserved variables and a timestep, adjust
 the value of the total energy for each cell according to the specified cooling
 function. */
__global__ void cooling_kernel(Real *dev_conserved, int nx, int ny, int nz,
                               int n_ghost, int n_fields, Real dt, Real gamma,
                               cudaTextureObject_t coolTexObj,
                               cudaTextureObject_t heatTexObj);

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

/* \fn __device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t
 coolTexObj, cudaTextureObject_t heatTexObj)
 * \brief Uses texture mapping to interpolate Cloudy cooling/heating
          tables at z = 0 with solar metallicity and an HM05 UV background. */
__device__ Real Cloudy_cool(Real n, Real T, cudaTextureObject_t coolTexObj,
                            cudaTextureObject_t heatTexObj);

  #endif  // COOLING_GPU
#endif    // CUDA
