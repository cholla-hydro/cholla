#pragma once
#ifdef PARTICLES_GPU

#include "../global/global.h"
#include "../analysis/feedback_analysis.h"
#ifdef O_HIP
#include <hiprand.h>
#include <hiprand_kernel.h>
#else
#include <curand_kernel.h>
#include <curand.h>
#endif //O_HIP


namespace supernova {
     const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4, UNRES_ENERGY = 5;

     // supernova rate: 1SN / 100 solar masses per 40^4 kyr
     static const Real SNR=2.5e-7;
     static const Real ENERGY_PER_SN  = 1e51 / MASS_UNIT*TIME_UNIT*TIME_UNIT/LENGTH_UNIT/LENGTH_UNIT;
     static const Real MASS_PER_SN    = 10.0;   // 10 solarMasses per SN
     static const Real FINAL_MOMENTUM = 2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT; // 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
     static const Real MU     = 0.6;
     static const Real R_SH   = 0.0302;         // 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
     static const Real SN_ERA = 4.0e4;          // assume SN occur during first 40 Myr after cluster formation.


     extern curandStateMRG32k3a_t*  randStates;
     extern part_int_t n_states;
     extern Real t_buff, dt_buff;

     void initState(struct parameters *P, part_int_t n_local, Real allocation_factor = 1);
     Real Cluster_Feedback(Grid3D& G, FeedbackAnalysis& sn_analysis);
}
#endif //PARTICLES_GPU
