/*! \file
 *  Define machinery for accessing field information
 */

#include "field_info.h"

#include <string>
#include <vector>

#include "../utils/FrozenKeyIdxBiMap.h"
#include "grid_enum.h"

namespace
{  // stuff in an anonymous namespace is local to this file

struct PropPack {
  const char* name;
  field::Kind kind;
};

}  // anonymous namespace

/*! list of all field names
 *
 *  This must remain synchronized with grid_enum.h
 */
static constexpr PropPack pack_arr_[] = {{"density", field::Kind::HYDRO},        {"momentum_x", field::Kind::HYDRO},
                                         {"momentum_y", field::Kind::HYDRO},     {"momentum_z", field::Kind::HYDRO},
                                         {"Energy", field::Kind::HYDRO},

#ifdef SCALAR
  // Add scalars here, wrapped appropriately with ifdefs:
  #ifdef BASIC_SCALAR
                                         {"basic_scalar", field::Kind::SCALAR},
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
                                         {"HI_density", field::Kind::SCALAR},    {"HII_density", field::Kind::SCALAR},
                                         {"HeI_density", field::Kind::SCALAR},   {"HeII_density", field::Kind::SCALAR},
                                         {"HeIII_density", field::Kind::SCALAR}, {"e_density", field::Kind::SCALAR},
    #ifdef GRACKLE_METALS
                                         {"metal_density", field::Kind::SCALAR},
    #endif
  #endif

  #ifdef DUST
                                         {"dust_density", field::Kind::SCALAR},
  #endif  // DUST

#endif  // SCALAR

#ifdef MHD
                                         {"magnetic_x", field::Kind::MAGNETIC},  {"magnetic_y", field::Kind::MAGNETIC},
                                         {"magnetic_z", field::Kind::MAGNETIC},
#endif
#ifdef DE
                                         {"GasEnergy", field::Kind::HYDRO}
#endif
};

static constexpr int n_fields_ = static_cast<int>(sizeof(pack_arr_) / sizeof(PropPack));

static_assert(n_fields_ == grid_enum::num_fields, "pack_arr_ and grid_enum::num_fields are no longer synchronized");

FieldInfo FieldInfo::create()
{
  FieldInfo out;

  // convert n_fields to a vector of std::string
  std::vector<std::string> v;
  v.reserve(n_fields_);
  for (std::size_t i = 0; i < n_fields_; i++) {
    v.push_back(std::string(pack_arr_[i].name));
    switch (pack_arr_[i].kind) {
      case field::Kind::HYDRO:
        out.hydro_field_ids_.push_back(i);
        break;
      case field::Kind::SCALAR:
        out.scalar_field_ids_.push_back(i);
        break;
      case field::Kind::MAGNETIC:
        out.magnetic_field_ids_.push_back(i);
        break;
      default:
        CHOLLA_ERROR("This branch should be unreachable");
    }
  }
  out.name_id_bimap_ = utils::FrozenKeyIdxBiMap(v);
  return out;
}

const std::vector<int>& FieldInfo::get_kind_ids_(field::Kind kind) const
{
  switch (kind) {
    case field::Kind::HYDRO:
      return hydro_field_ids_;
    case field::Kind::SCALAR:
      return scalar_field_ids_;
    case field::Kind::MAGNETIC:
      return magnetic_field_ids_;
    default:
      CHOLLA_ERROR("This branch should be unreachable");
  }
}