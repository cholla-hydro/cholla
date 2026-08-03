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
  field::IOBuf io_buf;
};

}  // anonymous namespace

/*! list of all field names
 *
 *  This must remain synchronized with grid_enum.h
 */
static constexpr PropPack pack_arr_[] = {
    {"density", field::Kind::HYDRO, field::IOBuf::DEVICE},
    {"momentum_x", field::Kind::HYDRO, field::IOBuf::DEVICE},
    {"momentum_y", field::Kind::HYDRO, field::IOBuf::DEVICE},
    {"momentum_z", field::Kind::HYDRO, field::IOBuf::DEVICE},
    {"Energy", field::Kind::HYDRO, field::IOBuf::DEVICE},

#ifdef SCALAR
  #ifdef BASIC_SCALAR
    // we use the name "scalar0" for better consistency with the name recorded during IO
    {"scalar0", field::Kind::PASSIVE_SCALAR, field::IOBuf::DEVICE},
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
    {"HI_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    {"HII_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    {"HeI_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    {"HeII_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    {"HeIII_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    {"e_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    #ifdef GRACKLE_METALS
    {"metal_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::HOST},
    #endif
  #endif

  #ifdef DUST
    {"dust_density", field::Kind::PASSIVE_SCALAR, field::IOBuf::DEVICE},
  #endif  // DUST

#endif  // SCALAR

#ifdef MHD
    {"magnetic_x", field::Kind::MAGNETIC, field::IOBuf::DEVICE},
    {"magnetic_y", field::Kind::MAGNETIC, field::IOBuf::DEVICE},
    {"magnetic_z", field::Kind::MAGNETIC, field::IOBuf::DEVICE},
#endif
#ifdef DE
    {"GasEnergy", field::Kind::HYDRO, field::IOBuf::DEVICE}
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
    v.emplace_back(pack_arr_[i].name);
    out.io_buf_.push_back(pack_arr_[i].io_buf);
    switch (pack_arr_[i].kind) {
      case field::Kind::HYDRO:
        out.hydro_field_ids_.push_back(i);
        break;
      case field::Kind::PASSIVE_SCALAR:
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
    case field::Kind::PASSIVE_SCALAR:
      return scalar_field_ids_;
    case field::Kind::MAGNETIC:
      return magnetic_field_ids_;
    default:
      CHOLLA_ERROR("This branch should be unreachable");
  }
}