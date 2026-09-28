/*! \file
 *  Define machinery for accessing field information
 */

#include "field_info.h"

#include <string>
#include <vector>

#include "../grid/grid_enum.h"
#include "../utils/FrozenKeyIdxBiMap.h"

namespace
{  // stuff in an anonymous namespace is local to this file

struct PropPack {
  const char* name;
  field::Kind kind;
  MemSpace io_buf;
};

}  // anonymous namespace

/*! list of all field names
 *
 *  This must remain synchronized with grid_enum.h
 */
static constexpr PropPack pack_arr_[] = {
    {"density", field::Kind::HYDRO, MemSpace::DEV},
    {"momentum_x", field::Kind::HYDRO, MemSpace::DEV},
    {"momentum_y", field::Kind::HYDRO, MemSpace::DEV},
    {"momentum_z", field::Kind::HYDRO, MemSpace::DEV},
    {"Energy", field::Kind::HYDRO, MemSpace::DEV},

#ifdef SCALAR
  #ifdef BASIC_SCALAR
    // we use the name "scalar0" for better consistency with the name recorded during IO
    {"scalar0", field::Kind::PASSIVE_SCALAR, MemSpace::DEV},
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
    {"HI_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    {"HII_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    {"HeI_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    {"HeII_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    {"HeIII_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    {"e_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    #ifdef GRACKLE_METALS
    {"metal_density", field::Kind::PASSIVE_SCALAR, MemSpace::HOST},
    #endif
  #endif

  #ifdef DUST
    {"dust_density", field::Kind::PASSIVE_SCALAR, MemSpace::DEV},
  #endif  // DUST

#endif  // SCALAR

#ifdef MHD
    {"magnetic_x", field::Kind::MAGNETIC, MemSpace::DEV},
    {"magnetic_y", field::Kind::MAGNETIC, MemSpace::DEV},
    {"magnetic_z", field::Kind::MAGNETIC, MemSpace::DEV},
#endif
#ifdef DE
    {"GasEnergy", field::Kind::HYDRO, MemSpace::DEV}
#endif
};

static constexpr int n_fields_ = static_cast<int>(sizeof(pack_arr_) / sizeof(PropPack));

static_assert(n_fields_ == grid_enum::num_fields, "pack_arr_ and grid_enum::num_fields are no longer synchronized");

field::IdRange FieldInfo::get_id_range(field::Kind kind) const
{
  using namespace std::literals;
  std::optional<uint8_t> tmp = pack_id("conserved"sv);
  if (!tmp.has_value()) {  // if there isn't a conserved field pack, return an empty range
    return field::IdRange(0, field_detail::IdxSlc());
  }
  uint8_t conserved_pack_id = tmp.value();

  switch (kind) {
    case field::Kind::HYDRO: {
      field_detail::IdxSlc primary_slc(0, grid_enum::Energy + 1);
#ifdef DE
      field_detail::IdxSlc secondary_slc(grid_enum::GasEnergy, grid_enum::GasEnergy + 1);
      return field::IdRange(conserved_pack_id, primary_slc, secondary_slc);
#else
      return field::IdRange(conserved_pack_id, primary_slc);
#endif
    }
    case field::Kind::PASSIVE_SCALAR: {
#ifdef SCALAR
      return field::IdRange(conserved_pack_id, field_detail::IdxSlc(grid_enum::scalar, grid_enum::finalscalar_plus_1));
#else
      // when SCALAR isn't defined, this range has no entries
      return field::IdRange(conserved_pack_id, field_detail::IdxSlc());
#endif
    }
    case field::Kind::MAGNETIC: {
#ifdef MHD
      static_assert(grid_enum::magnetic_x + 1 == grid_enum::magnetic_y, "bfields have unexpected order");
      static_assert(grid_enum::magnetic_y + 1 == grid_enum::magnetic_z, "bfields have unexpected order");
      return field::IdRange(conserved_pack_id, field_detail::IdxSlc(grid_enum::scalar, grid_enum::finalscalar_plus_1));
#else
      // when MHD isn't defined, this has no entries
      return field::IdRange(conserved_pack_id, field_detail::IdxSlc());
    }
#endif
      default:
        CHOLLA_ERROR("This branch should be unreachable");
    }
  }

  FieldInfo FieldInfo::create()
  {
    FieldInfo out;

    // here we construct 2 vectors that exist just as we initialize FieldInfo
    std::vector<std::string> pack_names{};
    pack_names.reserve(1);
    std::vector<std::string> flat_field_names;
    flat_field_names.reserve(n_fields_);

    // now we'll handle each each field-pack
    // (currently there's just 1 field-pack)

    // handle the conserved field pack
    // -------------------------------
    int n_conserved_fields = n_fields_;
    // -> enroll relevant field-pack information
    pack_names.emplace_back("conserved");
    field_detail::PackInfo conserved_pack_info{
        field_detail::IdxSlc(flat_field_names.size(), flat_field_names.size() + n_conserved_fields)};
    out.pack_info_.emplace_back(std::move(conserved_pack_info));

    // -> enroll info about each conserved field
    for (std::size_t i = 0; i < n_conserved_fields; i++) {
      flat_field_names.emplace_back(pack_arr_[i].name);
      out.io_buf_.push_back(pack_arr_[i].io_buf);
      FieldId field_id(0, static_cast<uint8_t>(i));
    }

    // now that we're done with all field packs, let's finalize things
    out.pname_id_bimap_ = utils::FrozenKeyIdxBiMap(pack_names);
    out.name_id_bimap_  = utils::FrozenKeyIdxBiMap(flat_field_names);
    return out;
  }