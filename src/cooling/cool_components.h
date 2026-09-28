/*! \file
 *  \brief Declarations of cooling functions. */

#pragma once

#include <cmath>

#include "../global/global.h"
#include "../io/ParameterMap.h"
#include "../utils/error_handling.h"
#include "../utils/gpu.hpp"  // inlcudes HIP header that define __forceinline__

/*! @defgroup coolcomp Cooling Component Logic
 *
 *  The goal here is to collect logic for modelling cooling. Cooling recipes are
 *  subsequently constructed from the one or more components.
 */
/** @{ */

/*! \brief Describes cooling components
 *
 *  \note Maybe we should call this namespace edot_component? (Since there's heating & cooling)
 *
 *  Each cooling recipe is composed of one or more cooling components. A cooling
 *  component is a "callable" (i.e. it's a function or it's a class that implements
 *  the appropriate method that let's be called just like a function).
 *
 *  Optimization Notes
 *  ==================
 *  It's important that that "core-logic" of each cooling component is implemented in a
 *  header file or in the same source file where a recipe actually invokes the
 *  "core logic". The "core-logic" includes code paths accessible through function-call
 *  syntax. To be more explicit,
 *  - when a cool-component is a regular function, the "core-logic" includes the whole
 *    function itself and any (user-defined) functions called by that function
 *  - for a callable class, the "core-logic" includes the definition of the
 *    `Real operator()(args...)` member function and any (user-defined) functions
 *    called by that member-function
 *
 *  We may also want to consider using ``__forceinline__``. With that said, we should
 *  only use ``__forceinline__`` if it explicitly provides a performance improvement
 *  (blindly using ``__forceinline__`` can actually hurt performance)
 *
 *  In the future, if most heating and cooling contributiond need ``log10(n)`` and
 *  ``log10(T)``, we may want to consider pre-computing those values. In this scenario,
 *  we probably want to continue providing ``n`` and ``T``. This could potentially
 *  improve performance in recipies using multiple independent contributions since
 *  ``log10`` and ``pow`` are generally a lot more expensive than most other operations
 *  relevant for computing heating and cooling
 */
namespace cool_component
{

/*! Primordial hydrogen/helium cooling curve (derived according to Katz et al. 1996.) */
inline __device__ Real primordial_cool(Real n, Real T)
{
  Real n_h, Y, y, g_ff, cool;
  Real n_h0, n_hp, n_he0, n_hep, n_hepp, n_e, n_e_old;
  Real alpha_hp, alpha_hep, alpha_d, alpha_hepp, gamma_eh0, gamma_ehe0, gamma_ehep;
  Real le_h0, le_hep, li_h0, li_he0, li_hep, lr_hp, lr_hep, lr_hepp, ld_hep, l_ff;
  Real gamma_lh0, gamma_lhe0, gamma_lhep, e_h0, e_he0, e_hep, H;
  int heat_flag, n_iter;
  Real diff, tol;

  // set flag to 1 for photoionization & heating
  heat_flag = 0;

  // Real X = 0.76; //hydrogen abundance by mass
  Y = 0.24;  // helium abundance by mass
  y = Y / (4 - 4 * Y);

  // set the hydrogen number density
  n_h = n;

  // calculate the recombination and collisional ionization rates
  // (Table 2 from Katz 1996)
  alpha_hp   = (8.4e-11) * (1.0 / sqrt(T)) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7))));
  alpha_hep  = (1.5e-10) * (pow(T, (-0.6353)));
  alpha_d    = (1.9e-3) * (pow(T, (-1.5))) * exp(-470000.0 / T) * (1.0 + 0.3 * exp(-94000.0 / T));
  alpha_hepp = (3.36e-10) * (1.0 / sqrt(T)) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7))));
  gamma_eh0  = (5.85e-11) * sqrt(T) * exp(-157809.1 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  gamma_ehe0 = (2.38e-11) * sqrt(T) * exp(-285335.4 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  gamma_ehep = (5.68e-12) * sqrt(T) * exp(-631515.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5)));
  // externally evaluated integrals for photoionization rates
  // assumed J(nu) = 10^-22 (nu_L/nu)
  gamma_lh0  = 3.19851e-13;
  gamma_lhe0 = 3.13029e-13;
  gamma_lhep = 2.00541e-14;
  // externally evaluated integrals for heating rates
  e_h0  = 2.4796e-24;
  e_he0 = 6.86167e-24;
  e_hep = 6.21868e-25;

  // assuming no photoionization, solve equations for number density of
  // each species
  n_e    = n_h;  // as a first guess, use the hydrogen number density
  n_iter = 20;
  diff   = 1.0;
  tol    = 1.0e-6;
  if (heat_flag) {
    for (int i = 0; i < n_iter; i++) {
      n_e_old = n_e;
      n_h0    = n_h * alpha_hp / (alpha_hp + gamma_eh0 + gamma_lh0 / n_e);
      n_hp    = n_h - n_h0;
      n_hep   = y * n_h /
              (1.0 + (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e) +
               (gamma_ehep + gamma_lhep / n_e) / alpha_hepp);
      n_he0  = n_hep * (alpha_hep + alpha_d) / (gamma_ehe0 + gamma_lhe0 / n_e);
      n_hepp = n_hep * (gamma_ehep + gamma_lhep / n_e) / alpha_hepp;
      n_e    = n_hp + n_hep + 2 * n_hepp;
      diff   = fabs(n_e_old - n_e);
      if (diff < tol) {
        break;
      }
    }
  } else {
    n_h0   = n_h * alpha_hp / (alpha_hp + gamma_eh0);
    n_hp   = n_h - n_h0;
    n_hep  = y * n_h / (1.0 + (alpha_hep + alpha_d) / (gamma_ehe0) + (gamma_ehep) / alpha_hepp);
    n_he0  = n_hep * (alpha_hep + alpha_d) / (gamma_ehe0);
    n_hepp = n_hep * (gamma_ehep) / alpha_hepp;
    n_e    = n_hp + n_hep + 2 * n_hepp;
  }

  // using number densities, calculate cooling rates for
  // various processes (Table 1 from Katz 1996)
  le_h0   = (7.50e-19) * exp(-118348.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_h0;
  le_hep  = (5.54e-17) * pow(T, (-0.397)) * exp(-473638.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_hep;
  li_h0   = (1.27e-21) * sqrt(T) * exp(-157809.1 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_h0;
  li_he0  = (9.38e-22) * sqrt(T) * exp(-285335.4 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_he0;
  li_hep  = (4.95e-22) * sqrt(T) * exp(-631515.0 / T) * (1.0 / (1.0 + sqrt(T / 1e5))) * n_e * n_hep;
  lr_hp   = (8.70e-27) * sqrt(T) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7)))) * n_e * n_hp;
  lr_hep  = (1.55e-26) * pow(T, (0.3647)) * n_e * n_hep;
  lr_hepp = (3.48e-26) * sqrt(T) * pow((T / 1e3), (-0.2)) * (1.0 / (1.0 + pow((T / 1e6), (0.7)))) * n_e * n_hepp;
  ld_hep  = (1.24e-13) * pow(T, (-1.5)) * exp(-470000.0 / T) * (1.0 + 0.3 * exp(-94000.0 / T)) * n_e * n_hep;
  g_ff    = 1.1 + 0.34 * exp(-(5.5 - log(T)) * (5.5 - log(T)) / 3.0);  // Gaunt factor
  l_ff    = (1.42e-27) * g_ff * sqrt(T) * (n_hp + n_hep + 4 * n_hepp) * n_e;

  // calculate total cooling rate (erg s^-1 cm^-3)
  cool = le_h0 + le_hep + li_h0 + li_he0 + li_hep + lr_hp + lr_hep + lr_hepp + ld_hep + l_ff;

  // calculate total photoionization heating rate
  H = 0.0;
  if (heat_flag) {
    H = n_h0 * e_h0 + n_he0 * e_he0 + n_hep * e_hep;
  }

  cool -= H;

  return cool;
}

/*! \brief Cooling the cooling function from Creasey 2011.
 *
 *  This was historically used as a test function (it isn't currently used for anything)
 *
 *  \return The cooling rate, lambda, in units of erg s^-1 cm^3 (it is NEVER negative)
 */
inline __device__ Real analytic_creasey11_lambda(Real n, Real T)
{
  Real T0 = 10000.0;
  Real T1 = 20 * T0;
  // Real lambda = 5.0e-24; //cooling coefficient, 5e-24 erg cm^3 s^-1
  Real lambda = 5.0e-20;  // cooling coefficient, 5e-24 erg cm^3 s^-1

  // constant cooling rate
  // cool = n*n*lambda;

  // Creasey cooling function
  if (T >= T0 && T <= 0.5 * (T1 + T0)) {
    return lambda * (T - T0) / T0;
  } else if (T >= 0.5 * (T1 + T0) && T <= T1) {
    return lambda * (T1 - T) / T0;
  } else {
    return 0.0;
  }
}

/*! \brief computes the cooling rate, based on an analytic fit to a solar metallicity
 *     CIE cooling curve calculated using Cloudy. For log10T, this returns 0
 *
 *   \return The cooling rate, lambda, in units of erg s^-1 cm^3 (it is NEVER negative)
 *
 *   \note
 *   It may not be necessary to use __forceinline__, I just used it to ensure I didn't harm existing
 *   performance
 *
 *   \note
 *   The actual formula for the fit is first described in the appendix of
 *   (Schneider & Robertson 2018)[https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S/abstract
 */
__device__ __forceinline__ Real analytic_cie_lambda(Real log10T)
{
  // fit to CIE cooling function
  if (log10T < 4.0) {
    return 0.0;
  } else if (log10T >= 4.0 && log10T < 5.9) {
    return pow(10.0, (-1.3 * (log10T - 5.25) * (log10T - 5.25) - 21.25));
  } else if (log10T >= 5.9 && log10T < 7.4) {
    return pow(10.0, (0.7 * (log10T - 7.1) * (log10T - 7.1) - 22.8));
  } else {
    return pow(10.0, (0.45 * log10T - 26.065));
  }
}

/*! \brief computes the cooling rate, based on an analytic fit provided in Koyama & Inutsuka (2002)
 *
 *  \return The cooling rate, lambda, in units of erg s^-1 cm^3 (it is NEVER negative)
 *
 *  \note
 *  It may not be necessary to use __forceinline__, I just used it to ensure I didn't harm existing
 *  performance
 *
 *  \note
 *  The actual formula for the fit is given as equations 4 and 5 in
 *  (Koyama & Inutsuka 2002)[https://ui.adsabs.harvard.edu/abs/2018ApJ...860..135S/abstract].
 */
__device__ __forceinline__ Real analytic_koyama_inutsuka_02_lambda(Real T)
{
  return 2e-26 * (1e7 * exp(-1.148e5 / (T + 1000.0)) + 1.4e-2 * sqrt(T) * exp(-92.0 / T));
}

/*! \brief Analytic cooling function recipe that roughly matches the "TI" cooling runs shown in
 *     in [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
 *
 *  For temperatures below 1e4 K:
 *  - We adopt the same analytic fitting formula as Kim & Ostriker 2015 for T < 1e4 K, which is an
 *    analytic fit to the results of Koyama & Inutsuka (2002).
 *  - a description of this fit is provided within
 *    [Kim+2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...681.1148K/abstract)
 *  For temperatures above 1e4 K
 *  - we directly use the exact same analytic CIE fit as CoolRecipeCIE
 *
 *  \return The cooling rate, lambda, in units of erg s^-1 cm^3 (it is NEVER negative)
 *
 *  \note
 *  It may not be necessary to use __forceinline__, I just used it to ensure I didn't harm existing
 *  performance
 *
 * \warning
 * Be aware, that all of our cooling infrastructure probably does not properly account for changes in
 * mean molecular weights. Historically, we just assumed a fixed mean molecular weight of 0.6 when we
 * used a CIE analytic fit. In practice, the fit below 1e4 K is intended to be used with a mean
 * molecular weight fixed to ~1.25
 */
__device__ __forceinline__ Real combined_analytic_ti_cie_lambda(Real T)
{
  if (T < 10.0) {
    return 0.0;  // no cooling below 10 K
  } else if (T >= 10.0 && T < 1e4) {
    return analytic_koyama_inutsuka_02_lambda(T);
  } else {
    return analytic_cie_lambda(log10(T));
  }
}

/*! Encapsulates our model and configuration for photoelectric heating
 *
 *  This implements a very simple model
 *  - we apply uniform photoelectric heating (over all space and time) to all gas at temperatures
 *    below 1e4 K
 *  - this model is described within
 *    [Kim & Ostriker 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...802...99K/abstract)
 *
 *  @note
 *  In the future, one could imagine implementing a more sophisticated model like TIGRESS
 *  - For example the amount of heating could be coupled with the properties of clusters
 *    within the simulation volume
 *  - If we started to model varying mmw, we could also adopt the TIGRESS strategy to more
 *    smoothly turn off heating at higher temperatures
 */
class PhotoelectricHeatingModel
{
  /*! This theoretically represents the mean density in the simulation volume. A value of 0.0
   *  indicates that there is no heating.
   *
   *  @note
   *  I can't remember the precise interpretation, but I think the idea may be that it may be
   *  used because it loosely relates to the rate of star formation...
   */
  double n_av_cgs_;

  inline static constexpr const char* use_photoelectric_parname  = "chemistry.photoelectric_heating";
  inline static constexpr const char* photoelectric_n_av_parname = "chemistry.photoelectric_n_av_cgs";

 public:
  __host__ static bool is_specified_by_params(ParameterMap& pmap)
  {
    return pmap.value_or(PhotoelectricHeatingModel::use_photoelectric_parname, false);
  }

  __host__ explicit PhotoelectricHeatingModel(ParameterMap& pmap)
  {
    // In this case, we want to actually use photoelectric heating
    if (pmap.value_or(PhotoelectricHeatingModel::use_photoelectric_parname, false)) {
      double n_av_cgs = pmap.value_or(PhotoelectricHeatingModel::photoelectric_n_av_parname, 100.0);
      CHOLLA_ASSERT(n_av_cgs > 0.0, "The \"%s\" parameter cannot specify a non-positive value",
                    PhotoelectricHeatingModel::photoelectric_n_av_parname);
      n_av_cgs_ = n_av_cgs;
    } else {
      // in this case, we initialize an instance that doesn't actually perform any
      // heating/cooling. We may want to get rid of this branch
      CHOLLA_ASSERT(!pmap.has_param(photoelectric_n_av_parname),
                    "It is an error to specify the \"%s\" parameter when the \"%s\" hasn't "
                    "explicitly been set to true.",
                    PhotoelectricHeatingModel::photoelectric_n_av_parname,
                    PhotoelectricHeatingModel::use_photoelectric_parname);
      n_av_cgs_ = 0.0;  // <- this means that there isn't heating
    }
  }

  bool is_active() const { return n_av_cgs_ != 0.0; }

  /*! \brief computes the heating rate per unit volume, erg /s / cm^3.
   *
   *  This **NEVER** returns a negative value.
   */
  __device__ Real operator()(Real n, Real T) const { return (T < 1e4) ? n * n_av_cgs_ * 1.0e-26 : 0.0; }
};

}  // namespace cool_component

/** @}*/  // end of group