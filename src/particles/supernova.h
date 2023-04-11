#pragma once
#if defined(PARTICLES_GPU) && defined(SUPERNOVA)

  #include "../analysis/feedback_analysis.h"
  #include "../global/global.h"
  #ifdef O_HIP
    #include <hiprand.h>
    #include <hiprand_kernel.h>
  #else
    #include <curand.h>
    #include <curand_kernel.h>
  #endif  // O_HIP

namespace supernova
{
const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4, UNRES_ENERGY = 5;

// supernova rate: 1SN / 100 solar masses per 36 Myr
static const Real DEFAULT_SNR   = 2.8e-7;
static const Real ENERGY_PER_SN = 1e51 / MASS_UNIT * TIME_UNIT * TIME_UNIT / LENGTH_UNIT / LENGTH_UNIT;
static const Real MASS_PER_SN   = 10.0;       // 10 solarMasses per SN
static const Real FINAL_MOMENTUM =
    2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT;    // 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
static const Real MU               = 0.6;
static const Real R_SH             = 0.0302;  // 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
static const Real DEFAULT_SN_END   = 40000;   // default value for when SNe stop (40 Myr)
static const Real DEFAULT_SN_START = 4000;    // default value for when SNe start (4 Myr)

void initState(struct parameters* P, part_int_t n_local, Real allocation_factor = 1);
Real Cluster_Feedback(Grid3D& G, FeedbackAnalysis& sn_analysis);
}  // namespace supernova
#endif  // PARTICLES_GPU && SUPERNOVA
