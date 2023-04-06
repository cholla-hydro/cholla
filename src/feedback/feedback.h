#pragma once
#if defined(PARTICLES_GPU) && defined(FEEDBACK)

  #include "../analysis/feedback_analysis.h"
  #include "../global/global.h"
  #ifdef O_HIP
    #include <hiprand.h>
    #include <hiprand_kernel.h>
  #else
    #include <curand.h>
    #include <curand_kernel.h>
  #endif  // O_HIP


typedef curandStateMRG32k3a_t feedback_prng_t;

namespace feedback
{
  const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4,
          UNRES_ENERGY = 5;

  // supernova rate: 1SN / 100 solar masses per 36 Myr
  static const Real DEFAULT_SNR = 2.8e-7;
  static const Real ENERGY_PER_SN =
    1e51 / MASS_UNIT * TIME_UNIT * TIME_UNIT / LENGTH_UNIT / LENGTH_UNIT;
  // 10 solarMasses per SN
  static const Real MASS_PER_SN = 10.0;
  // 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
  static const Real FINAL_MOMENTUM = 2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT;
  static const Real MU = 0.6;
  // 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
  static const Real R_SH = 0.0302;
  // default value for when SNe stop (40 Myr)
  static const Real DEFAULT_SN_END =  40000;
  // default value for when SNe start (4 Myr)
  static const Real DEFAULT_SN_START = 4000;

  extern feedback_prng_t*  randStates;
  extern part_int_t n_states;
  extern Real *dev_snr, snr_dt, time_sn_end, time_sn_start;
  extern Real *dev_sw_p, *dev_sw_e, sw_dt, time_sw_start, time_sw_end;


  #ifndef NO_SN_FEEDBACK
  void initState(struct parameters* P, part_int_t n_local,
                 Real allocation_factor = 1);
  #endif
  #ifndef NO_WIND_FEEDBACK
  void initWindState(struct parameters* P);
  #endif
  Real Cluster_Feedback(Grid3D& G, FeedbackAnalysis& sn_analysis);
}  // namespace supernova


#endif  // PARTICLES_GPU && FEEDBACK
