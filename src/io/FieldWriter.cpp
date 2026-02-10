/*!
 * \file
 * Implements the FieldWriter type
 */

#include "../io/FieldWriter.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../grid/field_info.h"
#include "../io/io.h"
#include "../utils/error_handling.h"

namespace io
{

namespace
{  // stuff inside an anonymous namespace is local to this file

// this is used to help setup both FieldWriter & F32FieldWriter
struct DsetSpecListBuilder_ {
  std::vector<io::DatasetSpecEntry>& vec;
  const FieldInfo& field_info;

  void add_entry(const char* name, WriteCond cond)
  {
    std::optional<int> maybe_field_id = field_info.field_id(name);
    if (!maybe_field_id.has_value()) {
      CHOLLA_ERROR("the current Cholla config has no \"%s\" field", name);
    }
    int field_id = maybe_field_id.value();
    vec.push_back({field_id, '/' + std::string(name), field_info.io_buf(field_id).value(), cond});
  }
};

}  // anonymous namespace

// todo: obviously we should move away from these compile-time ifdef statements
// -> I'm not so sure that we want to directly convert each compile-time conditions
//    to a runtime parameter
// -> instead, we may want to think about a set of choices that are more unified with
//    the Output_Float32 runtime-options (alternatively, we could alter the
//    Output_Float32 options)

#ifdef OUTPUT_MOMENTUM
static constexpr WriteCond MOMENTUM_CONDITION = WriteCond::ALWAYS;
#else
static constexpr WriteCond MOMENTUM_CONDITION = WriteCond::REQUIRE_COMPLETE_DATA;
#endif

#ifdef OUTPUT_ENERGY
static constexpr WriteCond ENERGY_CONDITION = WriteCond::ALWAYS;
#else
static constexpr WriteCond ENERGY_CONDITION = WriteCond::REQUIRE_COMPLETE_DATA;
#endif

#ifdef OUTPUT_METALS
static constexpr WriteCond METALS_CONDITION = WriteCond::ALWAYS;
#else
static constexpr WriteCond METALS_CONDITION = WriteCond::REQUIRE_COMPLETE_DATA;
#endif

// this may look a little funky, but it maintains historical behavior (i.e. e_density is
// always written for CHEMISTRY_GPU)
#if defined(OUTPUT_ELECTRONS) || defined(CHEMISTRY_GPU)
static constexpr WriteCond ELECTRONS_CONDITION = WriteCond::ALWAYS;
#else
static constexpr WriteCond ELECTRONS_CONDITION = WriteCond::REQUIRE_COMPLETE_DATA;
#endif

FieldWriter::FieldWriter(ParameterMap& pmap, const FieldInfo& field_info) : lazy_scratch_buf_(new LazyScratchBuf)
{
  // construct DsetSpecListBuilder_ to append entries to `this->h5_dataset_spec_.cc_dataset_entries`
  DsetSpecListBuilder_ dsentry_l_builder{this->h5_dataset_spec_.cc_dataset_entries, field_info};

  dsentry_l_builder.add_entry("density", WriteCond::ALWAYS);
  dsentry_l_builder.add_entry("momentum_x", MOMENTUM_CONDITION);
  dsentry_l_builder.add_entry("momentum_y", MOMENTUM_CONDITION);
  dsentry_l_builder.add_entry("momentum_z", MOMENTUM_CONDITION);
  dsentry_l_builder.add_entry("Energy", ENERGY_CONDITION);
#ifdef DE
  dsentry_l_builder.add_entry("GasEnergy", ENERGY_CONDITION);
#endif

  for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
    std::string name = field_info.field_name(field_id).value();
    if (name == "e_density") {
      dsentry_l_builder.add_entry(name.c_str(), ELECTRONS_CONDITION);
    } else if (name == "metal_density") {
      dsentry_l_builder.add_entry(name.c_str(), METALS_CONDITION);
    } else {
      dsentry_l_builder.add_entry(name.c_str(), WriteCond::ALWAYS);
    }
  }

#ifdef MHD
  h5_dataset_spec_.write_mag = {true, true, true};
#else
  h5_dataset_spec_.write_mag = {false, false, false};
#endif

  // For now, I'm intentionally ignoring the remaining assorted outputs (e.g.
  // temperature, gravitational potential). That stuff is still handled very manually)
}

F32FieldWriter::F32FieldWriter(ParameterMap& pmap, const FieldInfo& field_info) : lazy_scratch_buf_(new LazyScratchBuf)
{
  // construct DsetSpecListBuilder_ to append entries to `this->cc_dataset_entries`
  DsetSpecListBuilder_ dsentry_l_builder{this->dataset_spec_.cc_dataset_entries, field_info};

  // includes GasEnergy if applicable
  for (int field_id : field_info.get_id_range(field::Kind::HYDRO)) {
    std::string field_name = field_info.field_name(field_id).value();
    std::string param_name = "out_float32_" + field_name;
    if (pmap.value_or(param_name, 0)) {
      dsentry_l_builder.add_entry(field_name.c_str(), WriteCond::ALWAYS);
    }
  }

  auto map_name_to_idx = [](const std::string& field_name) -> int {
    switch (field_name[field_name.size() - 1]) {
      case 'x':
        return 0;
      case 'y':
        return 1;
      case 'z':
        return 2;
      default:
        CHOLLA_ERROR("unexepectedly received field_name: %s", field_name.c_str());
    }
  };

  for (int field_id : field_info.get_id_range(field::Kind::MAGNETIC)) {
    std::string field_name = field_info.field_name(field_id).value();
    if (pmap.value_or("out_float32_" + field_name, 0)) {
      this->dataset_spec_.write_mag[map_name_to_idx(field_name)] = true;
    }
  }
}

}  // namespace io