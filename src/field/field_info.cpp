/*! \file
 *  Define machinery for accessing field information
 */

#include "field_info.h"

#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "../grid/grid_enum.h"
#include "../utils/FrozenKeyIdxBiMap.h"

namespace
{  // stuff in an anonymous namespace is local to this file

/*! \brief specify all the input properties for a field pack
 *
 *  This information will be massaged/slightly reorganized when building FieldInfo
 */
struct FieldPackSpec {
  std::string pack_name;
  int ghost_depth;
  int n_dev_registers;
  int n_host_registers;
  std::vector<std::string> field_names;
  /// all fields can share an io_buf or each field can have a separate io_buf
  std::variant<MemSpace, std::vector<MemSpace>> io_buf;
  /// the following should only be constructed if the current object encodes
  /// information about the "conserved" field-pack
  std::optional<field_detail::ConservedIdRanges> conserved_id_ranges;
};

struct PropPack {
  const char* name;
  MemSpace io_buf;
};

}  // anonymous namespace

/*! list of all conserved field names
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

/*! \brief append a FieldPackSpec to the vector that describes the "conserved" field-pack
 */
static void add_conserved_pack_spec_(std::vector<FieldPackSpec>& pack_specs, int ghost_depth)
{
  uint8_t conserved_pack_id  = static_cast<uint8_t>(pack_specs.size());
  FieldPackSpec& pack_spec   = pack_specs.emplace_back();
  pack_spec.pack_name        = "conserved";
  pack_spec.ghost_depth      = ghost_depth;
  pack_spec.n_dev_registers  = 1;
  pack_spec.n_host_registers = 1;

  std::vector<MemSpace> io_bufs;
  for (int i = 0; i < n_fields_; i++) {
    auto [cur_field_name, cur_io_buf] = pack_arr_[i];
    pack_spec.field_names.push_back(cur_field_name);
    io_bufs.push_back(cur_io_buf);
  }
  pack_spec.io_buf = std::move(io_bufs);

  // finally let's move on to initializing pack_spec.conserved_id_ranges

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

  pack_spec.conserved_id_ranges = field_detail::ConservedIdRanges{hydro, scalar, magnetic};
}

static void add_gravity_pack_spec_(std::vector<FieldPackSpec>& pack_specs, int normal_ghost_depth)
{
  {
    FieldPackSpec& pack_spec   = pack_specs.emplace_back();
    pack_spec.pack_name        = "gravity_misc";
    pack_spec.ghost_depth      = normal_ghost_depth;
    pack_spec.n_host_registers = 1;
#if defined(GRAVITY_GPU)
    pack_spec.n_dev_registers = 1;
    pack_spec.io_buf          = MemSpace::DEV;  // <- not totally sure this is right...
#else
    pack_spec.n_dev_registers = 0;
    pack_spec.io_buf          = MemSpace::HOST;
#endif
    pack_spec.field_names = {"total_density"};
  }

  {
    FieldPackSpec& pack_spec = pack_specs.emplace_back();
    pack_spec.pack_name      = "gravity_potentials";
#ifdef N_GHOST_POTENTIAL
    pack_spec.ghost_depth = N_GHOST_POTENTIAL;
#else
    CHOLLA_ERROR("N_GHOST_POTENTIAL is not defined");
#endif
    pack_spec.n_host_registers = 1;
    pack_spec.n_dev_registers  = 1;
    pack_spec.io_buf           = MemSpace::DEV;
    pack_spec.field_names      = {
        "potential",
        "potential_1",
#ifdef GRAVITY_ANALYTIC_COMP
        "analytic_potential",
#endif
    };

    // quick note: if GRAVITY_GPU isn't defined, we historically would not define the
    // analytic_potential on the GPU. Rather than make another field-pack dedicated to
    // holding analytic_potential, I just decided to allocate the memory in this case
    // (I don't think this scenario ever comes up in practice)
  }
}

static void add_radiation_pack_spec_(std::vector<FieldPackSpec>& pack_specs, int normal_ghost_depth)
{
  {
    FieldPackSpec& pack_spec   = pack_specs.emplace_back();
    pack_spec.pack_name        = "radiation_source";
    pack_spec.ghost_depth      = normal_ghost_depth;
    pack_spec.n_host_registers = 0;
    pack_spec.n_dev_registers  = 1;
    pack_spec.io_buf           = MemSpace::DEV;
    pack_spec.field_names      = {"source"};
  }

  {
    FieldPackSpec& pack_spec   = pack_specs.emplace_back();
    pack_spec.pack_name        = "radiation_fields";
    pack_spec.ghost_depth      = normal_ghost_depth;
    pack_spec.n_host_registers = 1;
    pack_spec.n_dev_registers  = 1;
    pack_spec.io_buf           = MemSpace::DEV;
    pack_spec.field_names      = {"source"};
#ifdef RT_OTVET
    pack_spec.field_names = {"rf_intensity", "rf_HI_near", "rf_HeI_near", "rf_HeII_near",
                             "rf_HI_far",    "rf_HeI_far", "rf_HeII_far"};
#elif defined(RT_M1)
    pack_spec.field_names = {"rf_intensity", "rf_intensity_Mx", "rf_intensity_My", "rf_intensity_Mz",
                             "rf_HI",        "rf_HI_Mx",        "rf_HI_My",        "rf_HI_Mz",
                             "rf_HeI",       "rf_HeI_Mx",       "rf_HeI_My",       "rf_HeI_Mz",
                             "rf_HeII",      "rf_HeII_Mx",      "rf_HeII_My",      "rf_HeII_Mz"};
#else
    CHOLLA_ERROR("this function shouldn't be executed if not using otvet/m1 rt");
#endif
  }
}

FieldInfo FieldInfo::create(int normal_ghost_depth)
{
  CHOLLA_ASSERT(normal_ghost_depth >= 0, "ghost_depth can't be negative");

  // construct a vector of FieldPackSpec objects
  std::vector<FieldPackSpec> pack_specs;
  add_conserved_pack_spec_(pack_specs, normal_ghost_depth);
#if defined(GRAVITY)
  add_gravity_pack_spec_(pack_specs, normal_ghost_depth);
#endif

#if defined(RT)
  add_radiation_pack_spec_(pack_specs, normal_ghost_depth);
#endif

  // move onto build FieldInfo from pack_specs
  // -----------------------------------------
  FieldInfo out;

  std::vector<std::string> pack_names{};
  pack_names.reserve(pack_specs.size());
  std::vector<std::string> flat_field_names;
  flat_field_names.reserve(n_fields_);

  for (const FieldPackSpec& pack_spec : pack_specs) {
    std::size_t n_fields_in_pack = pack_spec.field_names.size();
    CHOLLA_ASSERT(n_fields_in_pack > 0, "a field-pack must have at least 1 field!");

    // enroll information about the field-pack as a whole
    pack_names.emplace_back(pack_spec.pack_name);
    field_detail::PackInfo cur_pack_info{
        field_detail::IdxSlc(flat_field_names.size(), flat_field_names.size() + n_fields_in_pack),
        pack_spec.ghost_depth, pack_spec.n_host_registers, pack_spec.n_dev_registers};
    out.pack_info_.emplace_back(cur_pack_info);

    if (pack_spec.pack_name == "conserved") {
      if (!pack_spec.conserved_id_ranges.has_value()) {
        CHOLLA_ERROR("conserved_id_ranges is missing!");
      }
      out.conserved_id_ranges_ = pack_spec.conserved_id_ranges;
    }

    // enroll information about individual fields in the field-pack flat_field_names
    flat_field_names.insert(flat_field_names.end(), pack_spec.field_names.begin(), pack_spec.field_names.end());

    // update out.io_buf_
    const MemSpace* common_io_buf = std::get_if<MemSpace>(&pack_spec.io_buf);
    if (common_io_buf != nullptr) {
      for (std::size_t i = 0; i < n_fields_in_pack; i++) {
        out.io_buf_.push_back(*common_io_buf);
      }
    } else {
      const std::vector<MemSpace>* io_buf_vec = std::get_if<std::vector<MemSpace>>(&pack_spec.io_buf);
      CHOLLA_ASSERT(io_buf_vec != nullptr, "sanity-check failure!");
      out.io_buf_.insert(out.io_buf_.end(), io_buf_vec->begin(), io_buf_vec->end());
    }

    CHOLLA_ASSERT(out.io_buf_.size() == flat_field_names.size(), "sanity-check failure!");
  }

  // now that we're done with all field packs, let's finalize things
  out.pname_id_bimap_ = utils::FrozenKeyIdxBiMap(pack_names);
  out.name_id_bimap_  = utils::FrozenKeyIdxBiMap(flat_field_names);
  return out;
}