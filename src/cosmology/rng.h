/*! \file
 *! \brief declares cosmology-related rng operations
 */

#pragma once
#ifdef O_HIP
  #include <hiprand/hiprand.h>
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand.h>
  #include <curand_kernel.h>
#endif  // O_HIP
#include "../global/global.h"

// typedef curandStatePhilox4_32_10_t rng_parallel_state_t;
typedef curandStateMRG32k3a_t rng_parallel_state_t;

#if PRECISION == 1
  #define gpurand_normal   curand_normal
  #define gpurand_normal4  curand_normal4
  #define gpurand_uniform2 curand_uniform2
  #define RNG_PI           3.14159265358979323846f
#endif
#if PRECISION == 2
  #define gpurand_normal   curand_normal_double
  #define gpurand_normal4  curand_normal4_double
  #define gpurand_uniform2 curand_uniform2_double
  #define RNG_PI           3.141592653589793238462643383279502884
#endif

/*! \brief Initialize a GPU-based RNG */
__global__ void RNG_Init_GPU(int nx_local, int ny_local, int nz_local, int nx_local_start, int ny_local_start,
                             int nz_local_start, int nx, int ny, int nz, uint64_t seed, rng_parallel_state_t *states);
__global__ void RNG_Init_TEST(int procID, int nx_local, int ny_local, int nz_local, int nx_local_start,
                              int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed,
                              rng_parallel_state_t *states);
__global__ void RNG_Normal_Field_GPU(Real *d_field, int nx_local, int ny_local, int nz_local, int nx_local_start,
                                     int ny_local_start, int nz_local_start, int nx, int ny, int nz, uint64_t seed,
                                     rng_parallel_state_t *states);
