/*! \file
 *  \brief Describes machinery for encoding properties about gas profiles in the disk
 *         problem.
 */

#pragma once

#include <optional>

#include "../../io/ParameterMap.h"
#include "../../utils/error_handling.h"

namespace galaxy_detail
{

/*! \brief encodes the kind of gas profile to use */
enum class GasProfileKind { ISOTHERMAL, ISENTROPIC };

/*! \brief nicely parse the gas profile kind */
inline GasProfileKind Parse_GasProfileKind(ParameterMap& pmap, const std::string& param_name,
                                           std::optional<GasProfileKind> dflt = std::nullopt)
{
  if (pmap.has_param(param_name)) {
    std::string tmp = pmap.value<std::string>(param_name);
    if (tmp == "isothermal") {
      return GasProfileKind::ISOTHERMAL;
    } else if (tmp == "isentropic") {
      return GasProfileKind::ISENTROPIC;
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

struct InitialDiskGasProps {
  GasProfileKind profile; /*!< Indicates whether to initialize an isothermal or isentropic disk */
  Real H_d;               /*!< initial guess at the scale-height (in kpc) */
  Real T_d;               /*!< gas temperature */

  explicit InitialDiskGasProps(ParameterMap& pmap)
      : profile{Parse_GasProfileKind(pmap, "model.galaxy.gas_disk.init.profile_kind")},
        H_d{pmap.value<double>("model.galaxy.gas_disk.init.initial_scale_height_guess_kpc")},
        T_d{pmap.value<double>("model.galaxy.gas_disk.init.temperature")}
  {
  }
};

}  // namespace galaxy_detail