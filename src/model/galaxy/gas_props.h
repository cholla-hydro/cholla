/*! \file
 *  \brief Describes machinery for encoding properties about gas profiles in the disk
 *         problem.
 */

#pragma once

#include <optional>

#include "../../global/global.h"
#include "../../io/ParameterMap.h"
#include "../../utils/error_handling.h"

namespace galaxy_detail
{

/*! \brief Describes target relationship b/t gas density and gas pressure for galaxy ICs
 *
 *  The gas distribution for a gravitationally bound stable hydrodynamic system can
 *  only be determined given a relationship that specifies the thermal pressure as a
 *  function of mass density (this is quite intuitive: you're balancing the pressure
 *  gradient against the gravitational force). Instances of this type, encode that
 *  relationship.
 *
 *  In more detail, an analytic relationship commonly used for this purpose is called
 *  a polytrope or `p=Kρᵐ`, where `m` is the polytropic index. At the time of writing,
 *  this type can represent 2 special cases of the polytrope:
 *  - isothermal, where `m=1` and all gas has a constant temperature. A user-provided
 *    temperature is used to specify the normalization, when coupled with the equation
 *    of state for the gas.
 *  - isentropic, where `m=ɣ` (`ɣ` is the adiabatic index). In this case, all gas has
 *    entropy or `p/(ρ^ɣ)`. A user-specified (temperature, mass-density) pair specifies
 *    the normalization, when coupled with the equation of state for the gas.
 */
class GasProfile
{
 public:
  enum class Kind { ISOTHERMAL, ISENTROPIC };

 private:  // attributes:
  Kind kind_;
  // the following values "anchor" the gas property profile. Values of 0 are used when
  // the variables aren't initialized
  double temperature_anchor_;
  double rho_anchor_Msun_per_kpc3_;

  static Kind parse_Kind_(ParameterMap& pmap, const std::string& param_name, std::optional<Kind> dflt)
  {
    if (pmap.has_param(param_name)) {
      std::string tmp = pmap.value<std::string>(param_name);
      if (tmp == "isothermal") {
        return Kind::ISOTHERMAL;
      } else if (tmp == "isentropic") {
        return Kind::ISENTROPIC;
      } else {
        CHOLLA_ERROR("only allowed options for %s param are isothermal and isentropic: found `%s`", param_name.c_str(),
                     tmp.c_str());
      }
    } else if (dflt.has_value()) {
      return dflt.value();
    } else {
      [[maybe_unused]] auto tmp = pmap.value<std::string>(param_name);  // <- aborts with informative error
      CHOLLA_ERROR("SHOULD BE UNREACHABLE");  // <- keeps compiler from warning about lack of return
    }
  }

 public:
  bool is_isentropic() const { return kind_ == Kind::ISENTROPIC; }
  bool is_isothermal() const { return kind_ == Kind::ISOTHERMAL; }
  Real temperature_anchor() const { return temperature_anchor_; }
  Real rho_anchor_Msun_per_kpc3() const
  {
    CHOLLA_ASSERT(kind_ != Kind::ISOTHERMAL, "shouldn't be called for isothermal profile");
    return rho_anchor_Msun_per_kpc3_;
  }

  // we delete default constructor to guarantee that all instances are fully valid
  GasProfile() = delete;

  /*! \brief Construct an object from a ParameterMap
   *
   *  The param_prefix argument is prepended to the names of parsed parameters. This
   *  allows us to reuse this parsing logic for different groups of parameters.
   */
  GasProfile(ParameterMap& pmap, const std::string& param_prefix, std::optional<Kind> dflt_kind = std::nullopt)
      : kind_{GasProfile::parse_Kind_(pmap, param_prefix + "profile_kind", dflt_kind)},
        temperature_anchor_{0.0},
        rho_anchor_Msun_per_kpc3_{0.0}
  {
    // parse T_name
    std::string T_name  = param_prefix + "T_anchor";
    temperature_anchor_ = pmap.value<double>(T_name);
    CHOLLA_ASSERT(temperature_anchor_ > 0, "%s must be positive: %g", T_name.c_str(), temperature_anchor_);

    // parse the value of rho_anchor_Msun_per_kpc3_
    std::string rho_cgs_name   = param_prefix + "rho_anchor_cgs";
    bool has_rho_cgs           = pmap.has_param(rho_cgs_name);
    std::string rho_codeU_name = param_prefix + "rho_anchor_Msun_per_kpc3";
    bool has_rho_codeU         = pmap.has_param(rho_codeU_name);

    if (has_rho_cgs and has_rho_codeU) {
      CHOLLA_ERROR("Can't specify %s & %s at the same time", rho_cgs_name.c_str(), rho_codeU_name.c_str());
    } else if (kind_ == Kind::ISOTHERMAL and (has_rho_cgs or has_rho_codeU)) {
      CHOLLA_ERROR("Can't specify %s or %s for isothermal profile", rho_cgs_name.c_str(), rho_codeU_name.c_str());
    } else if (has_rho_cgs) {
      double tmp = pmap.value<double>(rho_cgs_name);
      CHOLLA_ASSERT(tmp > 0, "%s must be positive: %g", rho_cgs_name.c_str(), tmp);
      rho_anchor_Msun_per_kpc3_ = static_cast<Real>(tmp / DENSITY_UNIT);
    } else if (has_rho_codeU) {
      rho_anchor_Msun_per_kpc3_ = pmap.value<double>(rho_codeU_name);
      CHOLLA_ASSERT(rho_anchor_Msun_per_kpc3_ > 0, "%s must be positive: %g", rho_codeU_name.c_str(),
                    rho_anchor_Msun_per_kpc3_);
    } else if (kind_ != Kind::ISOTHERMAL) {
      CHOLLA_ERROR("Either %s or %s is required for non-isothermal profile", rho_cgs_name.c_str(),
                   rho_codeU_name.c_str());
    }
  }
};

struct InitialDiskGasProps {
  GasProfile profile; /*!< Specifies the profile used for initializing gas */
  Real H_d;           /*!< initial guess at the scale-height (in kpc) */
  // in the future, we may also track an initial metallicity and information for
  // initializing turbulence

  explicit InitialDiskGasProps(ParameterMap& pmap)
      : profile(pmap, "model.galaxy.gas_disk.init."),
        H_d{pmap.value<double>("model.galaxy.gas_disk.init.initial_scale_height_guess_kpc")}
  {
    CHOLLA_ASSERT(H_d > 0, "initial_scale_height_guess of gas disk must be positive: %g", H_d);
  }
};

struct InitialCGMProps {
  GasProfile profile; /*!< Indicates gas profile of the CGM */
  Real R_anchor_kpc;  /*!< Spherical radius where the profile's anchoring conditions arise */

  // in the future, we may also track an initial metallicity and perhaps properties for
  // a rotating CGM

  explicit InitialCGMProps(ParameterMap& pmap)
      : profile(pmap, "model.galaxy.cgm_init.", std::make_optional(GasProfile::Kind::ISENTROPIC)),
        R_anchor_kpc{pmap.value<double>("model.galaxy.cgm_init.R_anchor_kpc")}
  {
    CHOLLA_ASSERT(R_anchor_kpc > 0, "R_anchor must be positive: %g", R_anchor_kpc);
    CHOLLA_ASSERT(not profile.is_isothermal(), "isothermal cgm not currently supported");
  }
};

}  // namespace galaxy_detail