#pragma once
#if defined(PARTICLES_GPU) && defined(FEEDBACK)

  #include "../analysis/feedback_analysis.h"
  #include "../global/global.h"
  #include "../feedback/ratecalc.h"

namespace feedback
{
const int SN = 0, RESOLVED = 1, NOT_RESOLVED = 2, ENERGY = 3, MOMENTUM = 4, UNRES_ENERGY = 5;

static const Real ENERGY_PER_SN = 1e51 / MASS_UNIT * TIME_UNIT * TIME_UNIT / LENGTH_UNIT / LENGTH_UNIT;
// 10 solarMasses per SN
static const Real MASS_PER_SN = 10.0;
// 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
static const Real FINAL_MOMENTUM = 2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT;
// 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
static const Real R_SH = 0.0302;
// default value for when SNe stop (40 Myr)
static const Real DEFAULT_SN_END = 40000;
// default value for when SNe start (4 Myr)
static const Real DEFAULT_SN_START = 4000;

extern Real *dev_snr, snr_dt, time_sn_end, time_sn_start;
extern Real *dev_sw_p, *dev_sw_e, sw_dt, time_sw_start, time_sw_end;

  #ifndef NO_SN_FEEDBACK
void Init_State(struct parameters* P);
  #endif
  #ifndef NO_WIND_FEEDBACK
void Init_Wind_State(struct parameters* P);
  #endif

struct ClusterFeedbackMethod {

  // at this precise moment this constructor should only be called after Init_State and
  // Init_Wind_State have been called (this is just a temporary quirk)
  ClusterFeedbackMethod(struct parameters& P, FeedbackAnalysis& analysis);

  /* Actually apply the stellar feedback (SNe and stellar winds) */
  void operator() (Grid3D& G);

private: // attributes

  FeedbackAnalysis& analysis;
  SNRateCalc snr_calc_;
  SWRateCalc sw_calc_;
};

}  // namespace feedback

#endif  // PARTICLES_GPU && FEEDBACK
