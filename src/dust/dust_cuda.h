#ifdef CUDA
  #ifdef DUST

    #ifndef DUST_CUDA_H
      #define DUST_CUDA_H

      #include <math.h>

      #include "../global/global.h"
      #include "../utils/gpu.hpp"

void Dust_Update(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt, Real gamma);

__global__ void Dust_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dt,
                            Real gamma);

__device__ __host__ Real calc_tau_sp(Real n, Real T);

__device__ __host__ Real calc_dd_dt(Real d_dust, Real tau_sp);

    #endif  // DUST
  #endif    // CUDA
#endif      // DUST_CUDA_H