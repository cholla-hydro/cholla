/*! /file global_cuda.h
 *  /brief Declarations of global variables and functions for the cuda kernels.
 */

#ifndef GLOBAL_CUDA_H
#define GLOBAL_CUDA_H

#ifdef CUDA

  #include <math.h>
  #include <stdio.h>
  #include <stdlib.h>

  #include "../global/global.h"
  #include "../utils/gpu.hpp"

  #define TPB 256  // threads per block
// #define TPB 64

extern bool memory_allocated;  // Flag becomes true after allocating the memory
                               // on the first timestep

// Arrays are global so that they can be allocated only once.
// Not all arrays will be allocated for every integrator
// GPU arrays
// conserved variables
extern Real *dev_conserved, *dev_conserved_half;
// input states and associated interface fluxes (Q* and F* from Stone, 2008)
// Note that for hydro the size of these arrays is n_fields*n_cells*sizeof(Real)
// while for MHD it is (n_fields-1)*n_cells*sizeof(Real), i.e. they has one
// fewer field than you would expect
extern Real *Q_Lx, *Q_Rx, *Q_Ly, *Q_Ry, *Q_Lz, *Q_Rz, *F_x, *F_y, *F_z;
// Constrained transport electric fields
extern Real *ctElectricFields;

// Arrays for potential in GPU: Will be set to NULL if not using GRAVITY
extern Real *dev_grav_potential;
extern Real *temp_potential;
extern Real *buffer_potential;

/*! \fn int sgn_CUDA
 *  \brief Mathematical sign function. Returns sign of x. */
__device__ inline int sgn_CUDA(Real x)
{
  if (x < 0) {
    return -1;
  } else {
    return 1;
  }
}

  // Define atomic_add if it's not supported
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
  #else
__device__ double atomicAdd(double *address, double val)
{
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
  #endif

  // This helper function exists to make it easier to find printfs inside
  // kernels
  #define kernel_printf printf

#endif  // GLOBAL_CUDA_H

#endif  // CUDA
