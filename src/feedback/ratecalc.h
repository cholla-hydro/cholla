#ifndef FEEDBACK_RATECALC_H
#define FEEDBACK_RATECALC_H

#ifdef O_HIP
  #include <hiprand.h>
  #include <hiprand_kernel.h>
#else
  #include <curand.h>
  #include <curand_kernel.h>
#endif  // O_HIP


#include <string>

#include "../global/global.h"

typedef curandStateMRG32k3a_t feedback_prng_t;

// This header declares classes that encapsulate calculations of SN rates and the rate of SW
// deposition
//
// Currently, they don't have destructors that deallocate the data. This is fine in the short term,
// since we only construct up to a single instance of each class on the host during the entire
// simulation. With that said, I do have a strategy in mind for resolving this.

// seed for poisson random number generator
#define FEEDBACK_SEED 42

// the starburst 99 total stellar mass input
// stellar wind momentum fluxes and SN rates
// must be divided by this to get per solar
// mass values.
#define S_99_TOTAL_MASS 1e6

namespace feedback{
/* The following should really be macros */
// supernova rate: 1SN / 100 solar masses per 36 Myr
static const Real DEFAULT_SNR   = 2.8e-7;
// default value for when SNe stop (40 Myr)
static const Real DEFAULT_SN_END = 40000;
// default value for when SNe start (4 Myr)
static const Real DEFAULT_SN_START = 4000;


/* Encapsulate Supernova Rate Calculation that is primarily intended to interpolate the data from
 * starburst99 tables.
 *
 * @note
 * The destructor doesn't currently deallocate the device heap-data. That's okay for the moment
 * because the only way to allocate that data at the moment is to call the table-reader constructor,
 * and that table-reader particular constructor is only called once during the entire duration of 
 * the simulation. With that said, I do have plans to address this issue in the future.
 */
struct SNRateCalc {

public:

  /* Default constructor. Ensures this object is always in a usable state
   *
   * This assumes a constant supernova rate given by feedback::DEFAULT_SNR
   */
  __host__ __device__ SNRateCalc()
    : dev_snr_(nullptr),
      snr_dt_(feedback::DEFAULT_SN_END - feedback::DEFAULT_SN_START),
      time_sn_start_(feedback::DEFAULT_SN_START),
      time_sn_end_(feedback::DEFAULT_SN_END)
  { }

  /* The "table-reader" constructor.
   *
   * Reads data from the specified file and allocates heapdata. If no file was specified, fall back
   * to configuration assumed in default constructor.
   *
   * @param P reference to parameters struct. Passes in starburst 99 filename.
   */
  __host__ SNRateCalc(struct parameters& P);

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
    // Note: in the C++ spec, wrap-around behavior is well-defined for unsigned types during integer
    //       overflow (overflow for signed types invokes undefined behavior)
    unsigned long long seed = (cluster_id < 0)
     ? (unsigned long long)(FEEDBACK_SEED) - (unsigned long long)(-1*cluster_id)
     : (unsigned long long)(FEEDBACK_SEED) + (unsigned long long)(cluster_id);
    curand_init(seed, 0, 0, &state);
    skipahead((unsigned long long)(n_step), &state);  // provided by curand
    return (int)curand_poisson(&state, ave_num_sn);
  }

  inline __device__ bool nonzero_sn_probability(Real age) const 
  {
    return (time_sn_start_ <= age) and (age <= time_sn_end_);
  }

private: // attributes
  /* device array with rate info */
  Real *dev_snr_;
  /* time interval between table data. Assumed to be constant. */
  Real snr_dt_;
  /* cluster age when SNR is first greater than zero. */
  Real time_sn_start_;
  /* cluster age when SNR drops to zero. */
  Real time_sn_end_;
};

/* Class responsible for computing stellar-wind rates
 *
 * @note
 * These were pretty much extracted directly from feedback.cu. There's a chance that there are some
 * logical errors in these functions. In particular, the way we have been using the Wind_Flux and
 * Wind_Power to update gas-momentum and gas-energy is inconsistent.
 *
 * @note
 * The destructor doesn't currently deallocate the device heap-data. That's okay for the moment
 * because the only way to allocate that data at the moment is to call the table-reader constructor,
 * and that table-reader particular constructor is only called once during the entire duration of 
 * the simulation. With that said, I do have plans to address this issue in the future.
 */
struct SWRateCalc {

  __host__ SWRateCalc(struct parameters& P);

  __host__ __device__ SWRateCalc(Real *dev_sw_p, Real* dev_sw_e, Real dt, Real t_start, Real t_end)
    : dev_sw_p_(dev_sw_p), dev_sw_e_(dev_sw_e), sw_dt_(dt), time_sw_start_(t_start), time_sw_end_(t_end)
  { }

  /* Get the Starburst 99 stellar wind momentum flux per solar mass.
   *
   * @param t cluster age in kyr
   * @return flux (in Cholla force units) per solar mass.
   */
  inline __device__ Real Get_Wind_Flux(Real t) const
  {
    if ((t < time_sw_start_) or (t >= time_sw_end_)) return 0;

    int index        = (int)((t - time_sw_start_) / sw_dt_);
    Real log_p_dynes = (dev_sw_p_[index] + (t - index * sw_dt_) *
                        (dev_sw_p_[index + 1] - dev_sw_p_[index]) / sw_dt_);
    return pow(10, log_p_dynes) / FORCE_UNIT / S_99_TOTAL_MASS;
  }

  /* Get the Starburst 99 stellar wind emitted power per solar mass.
   *
   * @param t cluster age in kyr
   * @return power (in Cholla units) per solar mass.
   */
  inline __device__ Real Get_Wind_Power(Real t) const
  {
    if ((t < time_sw_start_) or (t >= time_sw_end_)) return 0;

    int index  = (int)((t - time_sw_start_) / sw_dt_);
    Real log_e = (dev_sw_e_[index] + (t - index * sw_dt_) *
                  (dev_sw_e_[index + 1] - dev_sw_e_[index]) / sw_dt_);
    Real e     = pow(10, log_e) / (MASS_UNIT * VELOCITY_UNIT * VELOCITY_UNIT) * TIME_UNIT / S_99_TOTAL_MASS;
    return e;
  }

  /* Get the mass flux associated with stellar wind momentum flux and stellar wind power scaled per
   * cluster mass.
   *
   * @param flux
   * @return mass flux in g/s per solar mass
   */
  static __device__ Real Get_Wind_Mass(Real flux, Real power)
  {
    if ((flux <= 0) or (power <= 0)) return 0;
    return flux * flux / power / 2;
  }

  inline __device__ bool is_active(Real age) const
  {
    return (time_sw_start_ <= age) and (age <= time_sw_end_);
  }

private: // attributes
  /* device array of log base 10 momentum flux values in dynes. */
  Real *dev_sw_p_ = nullptr;
  /* device array of log base 10 power (erg/s) */
  Real *dev_sw_e_ = nullptr;
  /* time interval between table data points in kyr. */
  Real sw_dt_ = 0.0;
  /* cluster age when flux becomes non-negligible (kyr). */
  Real time_sw_start_ = 0.0;
  /* cluster age when stellar winds turn off (kyr). */
  Real time_sw_end_ = 0.0;
};

} // namespace

#endif /* FEEDBACK_RATECALC_H */