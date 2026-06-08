#include <string>

#include "../cooling/cooling_cuda.h"  // provides configure_cooling_callback
#include "../grid/grid3D.h"
#include "../io/ParameterMap.h"
#include "chemistry.h"

std::function<void(Grid3D&)> configure_chemistry_callback(ParameterMap& pmap)
{
  // we need to enforce this check so that we can accurately identify unnecessary parameters at the end of this func
  pmap.Enforce_Table_Content_Uniform_Access_Status("chemistry", true);

  // we use the traditional macros to set default-chemistry kinds to avoid
  // breaking older setups.
  // -> in the future, I think we can do away with this...
  std::string default_kind = "none";
#if defined(COOLING_GPU) && defined(CLOUDY_COOL)
  default_kind = "tabulated-cloudy";
#elif defined(COOLING_GPU)
  default_kind = "piecewise-cie";
#elif defined(CHEMISTRY_GPU)
  default_kind = "chemistry-gpu";
#elif defined(COOLING_GRACKLE)
  default_kind = "grackle";
#endif

  std::string chemistry_kind = pmap.value_or("chemistry.kind", default_kind);

  std::function<void(Grid3D&)> out{};

#if defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)
  CHOLLA_ASSERT(chemistry_kind == default_kind,
                "based on the defined macros, it is currently an error to pass a value to the "
                "chemistry.kind parameter other than \"%s\" (even \"none\" is invalid) This is "
                "because the \"%s\" functionality is invoked outside of the chemistry_callback "
                "machinery (this will be fixed in the near future)",
                default_kind.c_str(), default_kind.c_str());
#else
  if (chemistry_kind == "none") {
    // do nothing
  } else if (chemistry_kind == "chemistry-gpu" or chemistry_kind == "grackle") {
    CHOLLA_ERROR("chemistry.kind doesn't support %s yet (unless certain macros are defined)", chemistry_kind.c_str());
  } else if (chemistry_kind != "none") {
    out = configure_cooling_callback(chemistry_kind, pmap);
    if (not out) CHOLLA_ERROR("\"%s\" is not a supported chemistry.kind parameter value.", chemistry_kind.c_str());
  }
#endif  // defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)

  // ensure any errors if there are any parameters in the chemistry group that we have not accessed
  pmap.Enforce_Table_Content_Uniform_Access_Status("chemistry", false);
  return out;
}