#if defined(COSMOLOGY) && defined(PARTICLES_GPU)


#include "cosmology_functions_gpu.h"


__device__ Real Get_Hubble_Parameter_dev( Real a, Real H0, Real Omega_M, Real Omega_L, Real Omega_K ){
  Real a2 = a * a;
  Real a3 = a2 * a;
  Real factor = ( Omega_M/a3 + Omega_K/a2 + Omega_L );
  return H0 * sqrt(factor);
  
}


__device__ Real Get_Hubble_Parameter( Real a, Real H0, Real Omega_M, Real Omega_L ){
  Real a3 = a2 * a;
  Real factor = ( Omega_M/a3 + Omega_L );
  return H0 * sqrt(factor);
  
}



#endif //COSMOLOGY