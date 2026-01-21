/*! \file
 *  Define machinery for accessing field information
 */

#include "field_info.h"

#include <string>
#include <vector>

#include "../utils/FrozenKeyIdxBiMap.h"
#include "grid_enum.h"

/*! list of all field names
 *
 *  This must remain synchronized with grid_enum.h
 */
static const char* field_names_[] = {
    "density",       "momentum_x",  "momentum_y",  "momentum_z",   "Energy",

#ifdef SCALAR
  // Add scalars here, wrapped appropriately with ifdefs:
  #ifdef BASIC_SCALAR
    "basic_scalar",
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
    "HI_density",    "HII_density", "HeI_density", "HeII_density", "HeIII_density", "e_density",
    #ifdef GRACKLE_METALS
    "metal_density",
    #endif
  #endif

  #ifdef DUST
    "dust_density",
  #endif  // DUST

#endif  // SCALAR

#ifdef MHD
    "magnetic_x",    "magnetic_y",  "magnetic_z",
#endif
#ifdef DE
    "GasEnergy",
#endif
};

static constexpr std::size_t n_fields_ = sizeof(field_names_) / sizeof(const char*);

static_assert(n_fields_ == grid_enum::num_fields, "field_names_ and grid_enum::num_fields are no longer synchronized");

utils::FrozenKeyIdxBiMap get_field_id_mapping()
{
  // convert n_fields to a vector of std::string
  std::vector<std::string> v;
  v.reserve(n_fields_);
  for (std::size_t i = 0; i < n_fields_; i++) {
    v.push_back(std::string(field_names_[i]));
  }

  return utils::FrozenKeyIdxBiMap(v);
}
