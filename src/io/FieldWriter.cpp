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
  std::vector<io::DatasetSpecEntry>& vec = this->h5_dataset_spec_.cc_dataset_entries;
  auto add_dataset_entry                 = [&vec, &field_info](const char* name, WriteCond cond) {
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

#ifdef MHD
  h5_dataset_spec_.mhd_condition = std::optional<WriteCond>{WriteCond::REQUIRE_COMPLETE_DATA};
#else
  h5_dataset_spec_.mhd_condition = std::nullopt;
#endif

  // For now, I'm intentionally ignoring the remaining assorted outputs (e.g.
  // temperature, gravitational potential). That stuff is still handled very manually)
}

void FieldWriter::operator()(Grid3D& G, Parameters P, int nfile, const FnameTemplate& fname_template) const
{
  // create the filename
  std::string filename = fname_template.format_fname(nfile, "");

// open the file for binary writes
#ifdef HDF5
  hid_t file_id; /* file identifier */
  herr_t status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file
  G.Write_Grid_HDF5(file_id, h5_dataset_spec_);

  // close the file
  status = H5Fclose(file_id);

  if (status < 0) {
    printf("File write failed.\n");
    exit(-1);
  }

#else

  if (G.H.nx * G.H.ny * G.H.nz > 1000) printf("Ascii outputs only recommended for small problems!\n");
  // open the file for txt writes
  FILE* out;
  out = fopen(filename.data(), "w");
  if (out == NULL) {
    printf("Error opening output file.\n");
    exit(-1);
  }

  // write the header to the output file
  G.Write_Header_Text(out);

  // write the conserved variables to the output file
  G.Write_Grid_Text(out);

  // close the output file
  fclose(out);
#endif
}

}  // namespace io