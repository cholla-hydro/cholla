# pragma once
#ifdef O_HIP
  #include <hiprand.h>
  #include <hiprand_kernel.h>
#else
  #include <curand.h>
  #include <curand_kernel.h>
#endif  // O_HIP

typedef curandStatePhilox4_32_10_t rng_parallel_state_t;

#if PRECISION == 1
  #define gpurand_normal  curand_normal
#endif
#if PRECISION == 2
  #define gpurand_normal  curand_normal_double
#endif

namespace cholla_rng
{
}

__device__ void RNG_Init_GPU(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state);
inline __device__ void RNG_Normal_Field_GPU(Real *d_field, int n_cells, int n_ghost, curandStatePhilox4_32_10_t *state);
