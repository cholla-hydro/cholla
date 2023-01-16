#if defined(COSMOLOGY)

  #include "../global/global.h"
  #include "../grid/grid3D.h"
  #include "../io/io.h"
  #include "../utils/gpu.hpp"

  #define TPBX_COSMO 16
  #define TPBY_COSMO 8
  #define TPBZ_COSMO 8

// __device__ Real Get_Hubble_Parameter_dev( Real a, Real H0, Real Omega_M, Real
// Omega_L, Real Omega_K );

#endif  // COSMOLOGY
