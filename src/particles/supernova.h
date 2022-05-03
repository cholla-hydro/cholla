#ifndef SUPERNOVA_H
#define SUPERNOVA_H

#include "../global/global.h" 
#ifdef PARTICLES_GPU
#include <curand_kernel.h>
#include <curand.h>
#endif


namespace Supernova {
     const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4;
     
     // supernova rate: 1SN / 100 solar masses per 10^4 kyr
     static const Real SNR=1e-6;
     static const Real ENERGY_PER_SN = 5.3e-05;      // 1e51 ergs/SN in solarMass*(kpc/kyr)**2
     static const Real MASS_PER_SN   = 10.0;         // 10 solarMasses per SN
     static const Real FINAL_MOMENTUM = 0.29;        // 2.8e5 solarMasses km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
     static const Real MU = 0.6; 
     static const Real R_SH = 0.0302;                // 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
     static const Real SN_ERA = 1.0e4;               // assume SN occur during first 10 Myr after cluster formation.
    
     #ifdef PARTICLES_GPU
     extern curandStateMRG32k3a_t*  curandStates;
     extern part_int_t n_states;

     void initState(struct parameters *P, part_int_t n_local, Real allocation_factor = 1);
     //void initState(struct parameters *P);

     #endif //PARTICLES_GPU
}


#endif
