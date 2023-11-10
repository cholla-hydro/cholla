#ifndef FEEDBACK_RATECALC_H
#define FEEDBACK_RATECALC_H

#include "../feedback/feedback.h"

#ifdef O_HIP
  #include <hiprand.h>
  #include <hiprand_kernel.h>
#else
  #include <curand.h>
  #include <curand_kernel.h>
#endif  // O_HIP

typedef curandStateMRG32k3a_t feedback_prng_t;


// This declares classes that encapsulate calculations of SN rates and the rate of SW deposition
//
// Currently, they don't convey ownership over the required data. But the plan is to eventually
// add support for that

// seed for poisson random number generator
#define FEEDBACK_SEED 42

namespace feedback
{

struct SNRateCalc {

public:
  __host__ __device__ SNRateCalc(Real *dev_snr, Real dt, Real time_start, Real time_end)
    : dev_snr_(dev_snr), snr_dt_(dt), time_sn_start_(time_start), time_sn_end_(time_end) 
  { }

  /* returns supernova rate from starburst 99 (or default analytical rate).
   * 
   * Does a basic interpolation of S'99 table values.
   *
   * @param t   The cluster age.
   * @return number of SNe per kyr per solar mass
   *
   * @note
   * It's important to retain the inline annotation to maximize the chance of inlining.
   */
  inline __device__ Real Get_SN_Rate(Real t) const
  {
    if ((t < time_sn_start_) or (t >= time_sn_end_)) return 0;
    if (dev_snr_ == nullptr) return feedback::DEFAULT_SNR;

    int index = (int)((t - time_sn_start_) / snr_dt_);
    return dev_snr_[index] + (t - index * snr_dt_) * (dev_snr_[index + 1] - dev_snr_[index]) / snr_dt_;
  }

  /* Get an actual number of SNe given the expected number. Both the simulation step number
   * and cluster ID is used to set the state of the random number generator in a unique and
   * deterministic way.
   *
   * @param ave_num_sn expected number of SN, based on cluster age, mass and time step.
   * @param n_step sim step number
   * @param cluster_id
   * @return number of supernovae
   *
   * @note
   * It's important to retain the inline annotation to maximize the chance of inlining
   */
  static inline __device__ int Get_Number_Of_SNe_In_Cluster(Real ave_num_sn, int n_step,
                                                            part_int_t cluster_id)
  {
    feedback_prng_t state;
    curand_init(FEEDBACK_SEED, 0, 0, &state);
    unsigned long long skip = n_step * 10000 + cluster_id;
    skipahead(skip, &state);  // provided by curand
    return (int)curand_poisson(&state, ave_num_sn);
  }

  inline __device__ bool nonzero_sn_probability(Real age) const 
  {
    return (time_sn_start_ <= age) and (age <= time_sn_end_);
  }

private: // attributes
  /* device array with rate info */
  Real *dev_snr_ = nullptr;
  /* time interval between table data. Assumed to be constant. */
  Real snr_dt_ = 0.0;
  /* cluster age when SNR is first greater than zero. */
  Real time_sn_start_ = 0.0;
  /* cluster age when SNR drops to zero. */
  Real time_sn_end_ = 0.0;
};


} // namespace

#endif /* FEEDBACK_RATECALC_H */