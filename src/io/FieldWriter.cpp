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

FieldWriter::FieldWriter(ParameterMap& pmap, const FieldInfo& field_info)
{
  std::vector<io::DatasetSpec>& vec = this->h5_dataset_spec_;
  auto add_dataset_entry            = [&vec, &field_info](const char* name, WriteCond cond) {
    std::optional<int> maybe_field_id = field_info.field_id(name);
    if (!maybe_field_id.has_value()) {
      CHOLLA_ERROR("the current Cholla config has no \"%s\" field", name);
    }
    int field_id = maybe_field_id.value();
    vec.push_back({field_id, '/' + std::string(name), field_info.io_buf(field_id).value(), cond});
  };

  add_dataset_entry("density", WriteCond::ALWAYS);
  add_dataset_entry("momentum_x", MOMENTUM_CONDITION);
  add_dataset_entry("momentum_y", MOMENTUM_CONDITION);
  add_dataset_entry("momentum_z", MOMENTUM_CONDITION);
  add_dataset_entry("Energy", ENERGY_CONDITION);
#ifdef DE
  add_dataset_entry("GasEnergy", ENERGY_CONDITION);
#endif

  for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
    std::string name = field_info.field_name(field_id).value();
    if (name == "e_density") {
      add_dataset_entry(name.c_str(), ELECTRONS_CONDITION);
    } else if (name == "metal_density") {
      add_dataset_entry(name.c_str(), METALS_CONDITION);
    } else {
      add_dataset_entry(name.c_str(), WriteCond::ALWAYS);
    }
  }

  // For now, I'm intentionally ignoring the remaining assorted outputs (e.g. magnetic
  // fields, temperature, gravitational potential). That stuff is still handled very
  // manually)
}

}  // namespace io