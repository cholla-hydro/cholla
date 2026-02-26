/*!
 * \file
 * Implements the FieldWriter type
 */

#include "../io/FieldWriter.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
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

/*! this is used to help set up the `std::vector<io::DatasetSpecEntry>` that is tracked
 *  within @ref io::FieldWriter.
 *
 *  \note
 *  At this time, the `std::vector<io::DatasetSpecEntry>` is technically tracked by a
 *  @ref DatasetSpec that is tracked by @ref FieldWriter. But, in PR #469, we'll start
 *  tracking DatasetSpec directly as part of @ref FieldWriter
 *
 *  How it's Used
 *  =============
 *  An instance of this type is creatied during construction of @ref FieldWriter:
 *  - when an instance is maed, it wraps the vector that will be built-up and is
 *    configured with other information to properly append entries to the vector from
 *    field names.
 *  - then, the @ref add_entry method is called for each "normal" field that the writer
 *    will record in an output file (the instance will update the wrapped vector
 *    appropriately).
 *  - once we finish constructing @ref FieldWriter is constructed, the instance of this
 *    type is discarded
 */
struct DsetSpecListBuilder_ {
  using OutNameRecipie = std::function<std::string(std::string_view)>;

 private:
  /*! reference to the vector that the instance gradually "builds-up" */
  std::vector<io::DatasetSpecEntry>& wrapped_vec;
  /*! references the field_info from the current simulation. */
  const FieldInfo& field_info;
  /*! returns the output name used for a field based upon the original field name */
  OutNameRecipie out_name_recipe;
  /*! When specified, overides the typical choice associated with a field */
  std::optional<field::IOBuf> force_buf_choice;

 public:
  /*! make a new instance */
  DsetSpecListBuilder_(std::vector<io::DatasetSpecEntry>& wrapped_vec, const FieldInfo& field_info,
                       OutNameRecipie out_name_recipe, std::optional<field::IOBuf> force_buf_choice = std::nullopt)
      : wrapped_vec(wrapped_vec),
        field_info(field_info),
        out_name_recipe(out_name_recipe),
        force_buf_choice(force_buf_choice)
  {
    if (not out_name_recipe) CHOLLA_ERROR("passed an empty out_name_recipe");
  }

  void add_entry(const char* name, WriteCond cond)
  {
    // lookup the field_id associated with name
    std::optional<int> maybe_field_id = field_info.field_id(name);
    if (!maybe_field_id.has_value()) {
      CHOLLA_ERROR("the current Cholla config has no \"%s\" field", name);
    }
    int field_id = maybe_field_id.value();

    // determine the buffer that data will be written from
    field::IOBuf io_buf = force_buf_choice.has_value() ? force_buf_choice.value() : field_info.io_buf(field_id).value();

    // determine the dset_name
    std::string dset_name = out_name_recipe(name);

    wrapped_vec.emplace_back(field_id, std::move(dset_name), io_buf, cond);
  }
};

/*! Lookup the shortened output name that was historically used in text outputs
 *
 *  \todo Can we reuse this for a similar purpose when recording slices?
 *
 *  \note
 *  Earlier versions of the text output function were designed to shorten
 *  "magnetic_[xyz]" to "mag[XYZ]." But, as far as I can tell, this was never really
 *  used. Consequently, we don't shorten these field names.
 */
std::optional<std::string> lookup_legacy_short_name_(std::string_view field_name)
{
  if (field_name == "density") {
    return std::optional<std::string>("rho");
  } else if (field_name == "momentum_x") {
    return std::optional<std::string>("mx");
  } else if (field_name == "momentum_y") {
    return std::optional<std::string>("my");
  } else if (field_name == "momentum_z") {
    return std::optional<std::string>("mz");
  } else if (field_name == "Energy") {
    return std::optional<std::string>("E");
  } else if (field_name == "GasEnergy") {
    return std::optional<std::string>("ge");
  } else {
    return std::nullopt;
  }
}

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

FieldWriter::FieldWriter(FileFormat file_format, ParameterMap& pmap, const FieldInfo& field_info)
{
  this->file_format_ = file_format;

  // determine configuration parameters for DsetSpecListBuilder_
  std::optional<field::IOBuf> force_buf_choice = std::nullopt;
  std::function<std::string(std::string_view)> out_name_recipe;
  if (this->file_format_ == FileFormat::TEXT) {
    // when writing text outputs
    // -> we ALWAYS save data from host
    force_buf_choice = std::optional<field::IOBuf>{field::IOBuf::HOST};
    // -> we try to use the legacy short name, but if none exists, we reuse the field
    //    name as the output name
    out_name_recipe = [](std::string_view field_name) -> std::string {
      std::optional<std::string> maybe_out_name = lookup_legacy_short_name_(field_name);
      if (maybe_out_name.has_value()) return maybe_out_name.value();
      return std::string(field_name);
    };
  } else {  // this->file_format == FileFormat::H5_NATIVE_PRECISION
    out_name_recipe = [](std::string_view field_name) { return '/' + std::string(field_name); };
  }

  // construct DsetSpecListBuilder_
  std::vector<io::DatasetSpecEntry>& vec = this->dataset_spec_.cc_dataset_entries;
  DsetSpecListBuilder_ registrar(vec, field_info, out_name_recipe, force_buf_choice);

  registrar.add_entry("density", WriteCond::ALWAYS);
  registrar.add_entry("momentum_x", MOMENTUM_CONDITION);
  registrar.add_entry("momentum_y", MOMENTUM_CONDITION);
  registrar.add_entry("momentum_z", MOMENTUM_CONDITION);
  registrar.add_entry("Energy", ENERGY_CONDITION);
#ifdef DE
  registrar.add_entry("GasEnergy", ENERGY_CONDITION);
#endif

  for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
    std::string name = field_info.field_name(field_id).value();
    if (name == "e_density") {
      registrar.add_entry(name.c_str(), ELECTRONS_CONDITION);
    } else if (name == "metal_density") {
      registrar.add_entry(name.c_str(), METALS_CONDITION);
    } else {
      registrar.add_entry(name.c_str(), WriteCond::ALWAYS);
    }
  }

#ifdef MHD
  dataset_spec_.mhd_condition = std::optional<WriteCond>{WriteCond::REQUIRE_COMPLETE_DATA};
#else
  dataset_spec_.mhd_condition = std::nullopt;
#endif

  // For now, I'm intentionally ignoring the remaining assorted outputs (e.g.
  // temperature, gravitational potential). That stuff is still handled very manually)
}

/*! Helper function (for writing text output files) that setup for each relevant field
 *
 *  In more detail, this function gets called when creating a new text file field-dump.
 *  The function determines the fields that will be written out (based upon
 *  @p dataset_spec and the value of @p Output_Complete_Data ) and for each field:
 *  - it writes out the appropriate column names to @p fp
 *  - fills @p ptr_arr and @p is_cell_centered with appropriate values.
 *
 *  \param[out] ptr_arr Buffer filled with host-pointers for each relevant field
 *  \param[out] is_cell_centered Buffer that specifies whether the corresponding entry
 *      in @p ptr_arr corresponds to a face-centered or cell-centered field
 *  \param[out] fp  output file stream that column names are written to
 *  \param[in] dataset_spec Specifies properties about all fields that may be written
 *      by the current simulation
 *  \param[in] G specifies all grid data
 *
 *  \returns The number of entries written to ptr_arr
 */
static int Record_Colnames_And_Get_Field_Ptrs_(const Real** ptr_arr, bool* is_cell_centered, std::FILE* fp,
                                               const DatasetSpec& dataset_spec, const Grid3D& G)
{
  const Header& H             = G.H;
  const Grid3D::Conserved& C  = G.C;
  const FieldInfo& field_info = G.field_info;

  // write the name of the first column (the index column)
  std::fprintf(fp, "id");

  int field_ptr_counter = 0;

  // iterate over the cell-centered fields
  for (const io::DatasetSpecEntry& entry : dataset_spec.cc_dataset_entries) {
    // perform 2 simple sanity checks (these check invariants that should be satisfied
    // during initialization)
    CHOLLA_ASSERT(entry.io_buf == field::IOBuf::HOST, "io_buf sanity check failed!");
    CHOLLA_ASSERT(field_info.is_cell_centered(entry.field_id).value_or(false), "field-centering sanity-check failed!");

    if (entry.condition == io::WriteCond::REQUIRE_COMPLETE_DATA and not H.Output_Complete_Data) {
      continue;
    }

    std::fprintf(fp, "\t%s", entry.name.c_str());  // <- write the column name

    // record the field's pointer and whether it is cell-centered
    ptr_arr[field_ptr_counter]          = &G.C.host[entry.field_id * G.H.n_cells];
    is_cell_centered[field_ptr_counter] = true;
    field_ptr_counter++;
  }

  // now, let's handled magnetic fields (if applicable)
  if (dataset_spec.mhd_condition.has_value() and
      (dataset_spec.mhd_condition.value() == io::WriteCond::ALWAYS or H.Output_Complete_Data)) {
    const char* field_names[3] = {"magnetic_x", "magnetic_y", "magnetic_z"};
    for (const char* field_name : field_names) {
      std::fprintf(fp, "\t%s", field_name);  // <- write the column name

      const Real* ptr                     = &G.C.host[H.n_cells * field_info.field_id(field_name).value()];
      ptr_arr[field_ptr_counter]          = ptr;
      is_cell_centered[field_ptr_counter] = false;
      field_ptr_counter++;
    }
  }

  std::fputc('\n', fp);

  return field_ptr_counter;
}

/*! Helper function that write the conserved quantities to a text output file
 *
 *  \param[out] fp output file stream that column names are written to
 *  \param[in] G specifies all grid data
 *  \param[in] dataset_spec Specifies properties about all fields that may be written
 *      by the current simulation
 *
 *  \note
 *  The fact that the data is interleaved (and the fact that the number of characters
 *  per row is a variable), makes this a little tricky.
 */
static void Write_Grid_Text_(std::FILE* fp, const Grid3D& G, const DatasetSpec& dataset_spec)
{
  const Header& H             = G.H;
  const Grid3D::Conserved& C  = G.C;
  const FieldInfo& field_info = G.field_info;

  // sanity check: the factory method should prevent construction of FieldWriter in
  // circumstances where the following assertion would fail
  bool is_1D = (H.nx > 1 && H.ny == 1 && H.nz == 1);
  CHOLLA_ASSERT(is_1D, "can only write Fields to text files for 1D datasets");

  // Write the conserved quantities to the output file

  constexpr int MAX_FIELDS = 20;  // <- make this bigger, if necessary
  {                               // perform some sanity checks!
    const int n_fields = field_info.n_fields();
    CHOLLA_ASSERT(n_fields > 0, "n_names must be positive");
    CHOLLA_ASSERT(n_fields <= MAX_FIELDS, "n_fields %d exceeds MAX_FIELDS, %d", n_fields, MAX_FIELDS);
  }

  // Part 1: collect info about each field & write the initial header for the text file
  // ----------------------------------------------------------------------------------
  // these arrays will hold entries for each field that we want to serialize
  // (the precise details may depend upon G.H.Output_Complete_Data)
  const Real* ptr_arr[MAX_FIELDS];
  bool is_cell_centered[MAX_FIELDS];
  int n_output_fields = Record_Colnames_And_Get_Field_Ptrs_(ptr_arr, is_cell_centered, fp, dataset_spec, G);

  bool all_cell_centered = true;
  for (int ptr_idx = 0; ptr_idx < n_output_fields; ptr_idx++) {
    all_cell_centered = all_cell_centered or is_cell_centered[ptr_idx];
  }

  // Part 2: Record all data
  // -----------------------
  // write all normal rows
  for (int i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
    std::fprintf(fp, "%d", i - H.n_ghost);
    for (int ptr_idx = 0; ptr_idx < n_output_fields; ptr_idx++) {
      std::fprintf(fp, "\t%f", ptr_arr[ptr_idx][i]);
    }
    std::fputc('\n', fp);
  }

  // if any fields are not cell-centered (i.e. the magnetic fields), we need to write
  // an extra table row
  if (not all_cell_centered) {
    int i = H.nx - H.n_ghost;
    std::fprintf(fp, "%d", i - H.n_ghost);
    for (int ptr_idx = 0; ptr_idx < n_output_fields; ptr_idx++) {
      if (is_cell_centered[ptr_idx]) {
        std::fprintf(fp, "\tNaN");
      } else {
        std::fprintf(fp, "\t%f", ptr_arr[ptr_idx][i]);
      }
    }
    std::fputc('\n', fp);
  }
}

void FieldWriter::operator()(Grid3D& G, Parameters P, int nfile, const FnameTemplate& fname_template) const
{
  // create the filename
  std::string filename = fname_template.format_fname(nfile, "");

  if (this->file_format_ == FileFormat::H5_NATIVE_PRECISION) {
#ifndef HDF5
    CHOLLA_ERROR(
        "This error should never show up. Somehow FieldWriter was configured "
        "to write hdf5 files when cholla wasn't compiled with hdf5");
#else
    // Create a new file using default properties.
    hid_t file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write the header (file attributes)
    G.Write_Header_HDF5(file_id);

    // write the conserved variables to the output file
    G.Write_Grid_HDF5(file_id, this->dataset_spec_);

    // close the file
    if (H5Fclose(file_id) < 0) {
      CHOLLA_ERROR("File write failed.");
    }
#endif
  } else if (this->file_format_ == FileFormat::TEXT) {
    if (G.H.nx * G.H.ny * G.H.nz > 1000) printf("Ascii outputs only recommended for small problems!\n");
    // open the file for txt writes
    std::FILE* out = std::fopen(filename.data(), "w");
    if (out == nullptr) {
      CHOLLA_ERROR("Error opening output file.");
    }

    // write the header to the output file
    G.Write_Header_Text(out);

    // write the conserved variables to the output file
    Write_Grid_Text_(out, G, this->dataset_spec_);

    // close the output file
    std::fclose(out);
  } else {
    CHOLLA_ERROR("can't handle specified file format");
  }
}

}  // namespace io
