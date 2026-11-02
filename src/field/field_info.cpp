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
  MemSpace io_buf;
};

}  // anonymous namespace

/*! list of all field names
 *
 *  This must remain synchronized with grid_enum.h
 */
static constexpr PropPack pack_arr_[] = {
    {"density", MemSpace::DEV},
    {"momentum_x", MemSpace::DEV},
    {"momentum_y", MemSpace::DEV},
    {"momentum_z", MemSpace::DEV},
    {"Energy", MemSpace::DEV},

#ifdef SCALAR
  #ifdef BASIC_SCALAR
    // we use the name "scalar0" for better consistency with the name recorded during IO
    {"scalar0", MemSpace::DEV},
  #endif

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
    {"HI_density", MemSpace::HOST},
    {"HII_density", MemSpace::HOST},
    {"HeI_density", MemSpace::HOST},
    {"HeII_density", MemSpace::HOST},
    {"HeIII_density", MemSpace::HOST},
    {"e_density", MemSpace::HOST},
    #ifdef GRACKLE_METALS
    {"metal_density", MemSpace::HOST},
    #endif
  #endif

  #ifdef DUST
    {"dust_density", MemSpace::DEV},
  #endif  // DUST

#endif  // SCALAR

#ifdef MHD
    {"magnetic_x", MemSpace::DEV},
    {"magnetic_y", MemSpace::DEV},
    {"magnetic_z", MemSpace::DEV},
#endif
#ifdef DE
    {"GasEnergy", MemSpace::DEV}
#endif
};

static constexpr int n_fields_ = static_cast<int>(sizeof(pack_arr_) / sizeof(PropPack));

static_assert(n_fields_ == grid_enum::num_fields, "pack_arr_ and grid_enum::num_fields are no longer synchronized");

static field_detail::ConservedIdRanges build_ConservedIdRanges_(uint8_t conserved_pack_id)
{
  using field::IdRange;
  using field_detail::IdxSlc;

  // build the IdRange for hydro fields
  // ----------------------------------
#ifdef DE
  // when using dual-energy, the IdRange spans 2 separate index-slices
  IdRange hydro(conserved_pack_id, IdxSlc(0, grid_enum::Energy + 1),
                IdxSlc(grid_enum::GasEnergy, grid_enum::GasEnergy + 1));
#else
  // when not using dual-energy, the IdRange spans 2 separate index-slices
  IdRange hydro(conserved_pack_id, IdxSlc(0, grid_enum::Energy + 1));
#endif

  // build the IdRange for passive scalar fields
  // -------------------------------------------
#ifdef SCALAR
  IdRange scalar(conserved_pack_id, IdxSlc(grid_enum::scalar, grid_enum::finalscalar_plus_1));
#else
  // when SCALAR isn't defined, the range has no entries
  IdRange scalar(conserved_pack_id, IdxSlc());
#endif

  // build the IdRange for magnetic fields scalar fields
  // -------------------------------------------
#ifdef MHD
  static_assert(grid_enum::magnetic_x + 1 == grid_enum::magnetic_y, "bfields have unexpected order");
  static_assert(grid_enum::magnetic_y + 1 == grid_enum::magnetic_z, "bfields have unexpected order");
  IdRange magnetic(conserved_pack_id, IdxSlc(grid_enum::magnetic_x, grid_enum::magnetic_z + 1));
#else
  // when MHD isn't defined, the range has no entries
  IdRange magnetic(conserved_pack_id, field_detail::IdxSlc());
#endif

  return {hydro, scalar, magnetic};
}

FieldInfo FieldInfo::create(int normal_ghost_depth)
{
  CHOLLA_ASSERT(normal_ghost_depth >= 0, "ghost_depth can't be negative");
  FieldInfo out;

  // here we construct 2 vectors that exist just as we initialize FieldInfo
  std::vector<std::string> pack_names{};
  pack_names.reserve(1);
  std::vector<std::string> flat_field_names;
  flat_field_names.reserve(n_fields_);

  // now we'll handle each field-pack
  // (currently there's just 1 field-pack)

  // handle the conserved field pack
  // -------------------------------
  int n_conserved_fields = n_fields_;
  // -> enroll relevant field-pack information
  pack_names.emplace_back("conserved");
  field_detail::PackInfo conserved_pack_info{
      field_detail::IdxSlc(flat_field_names.size(), flat_field_names.size() + n_conserved_fields), normal_ghost_depth,
      1, 1};
  out.pack_info_.emplace_back(conserved_pack_info);

  // -> enroll info about each conserved field
  for (std::size_t i = 0; i < n_conserved_fields; i++) {
    flat_field_names.emplace_back(pack_arr_[i].name);
    out.io_buf_.push_back(pack_arr_[i].io_buf);
    FieldId field_id(0, static_cast<uint8_t>(i));
  }
  // construct the ConservedIdRanges so that we can easily iterate over subsets of the
  // conserved fields later
  uint8_t conserved_pack_id = static_cast<uint8_t>(pack_names.size() - 1);
  out.conserved_id_ranges_  = build_ConservedIdRanges_(conserved_pack_id);

  // now that we're done with all field packs, let's finalize things
  out.pname_id_bimap_ = utils::FrozenKeyIdxBiMap(pack_names);
  out.name_id_bimap_  = utils::FrozenKeyIdxBiMap(flat_field_names);
  return out;
}