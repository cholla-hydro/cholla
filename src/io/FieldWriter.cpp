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

// the following anonymous namespace encloses machinery used to help us implement
// FieldWriter's constructor
namespace
{  // stuff inside an anonymous namespace is local to this file

/*! this is used to help set up the `std::vector<io::DatasetSpecEntry>` that is tracked
 *  within @ref io::FieldWriter.
 *
 *  \note
 *  At this time, the `std::vector<io::DatasetSpecEntry>` is technically tracked by a
 *  @ref DatasetSpec that is tracked by @ref FieldWriter.
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
struct DsetSpecListBuilder {
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
  DsetSpecListBuilder(std::vector<io::DatasetSpecEntry>& wrapped_vec, const FieldInfo& field_info,
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

    // determine the dset_name
    std::string dset_name = out_name_recipe(name);

    if (force_buf_choice.has_value()) {
      wrapped_vec.emplace_back(field_id, std::move(dset_name), force_buf_choice.value(), cond);
    } else {
      std::optional<field::IOBuf> io_buf = field_info.io_buf(field_id);
      wrapped_vec.emplace_back(field_id, std::move(dset_name), get_or_abort(io_buf), cond);
    }
  }
};

int bfield_name_to_012_(std::string_view field_name)
{
  if (field_name == "magnetic_x") {
    return 0;
  } else if (field_name == "magnetic_y") {
    return 1;
  } else if (field_name == "magnetic_z") {
    return 2;
  } else {
    std::string tmp(field_name);
    CHOLLA_ERROR("unexepectedly received field_name: %s", tmp.c_str());
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
    : lazy_scratch_buf_(std::make_shared<LazyScratchBuf>())
{
  this->file_format_ = file_format;

  // Part 1: create a DsetSpecListBuilder instance
  // ==============================================

  // Part 1A: determine configuration parameters for DsetSpecListBuilder_
  std::optional<field::IOBuf> force_buf_choice = std::nullopt;
  std::function<std::string(std::string_view)> out_name_recipe;
  switch (this->file_format_) {
    case FileFormat::TEXT:
      force_buf_choice = std::optional<field::IOBuf>{field::IOBuf::HOST};
      // set output name to legacy short name (fall back to field name if there isn't a short name)
      out_name_recipe = [](std::string_view field_name) -> std::string {
        std::optional<std::string> maybe_out_name = lookup_legacy_short_name_(field_name);
        if (maybe_out_name.has_value()) return maybe_out_name.value();
        return std::string(field_name);
      };
      break;
    case FileFormat::H5_F32:
      force_buf_choice = std::optional<field::IOBuf>{field::IOBuf::DEVICE};
      [[fallthrough]];
    case FileFormat::H5_NATIVE_PRECISION:
      out_name_recipe = [](std::string_view field_name) { return '/' + std::string(field_name); };
      break;
    default:
      CHOLLA_ERROR("can't handle specified file format");
  }

  // Part 1B: actually DsetSpecListBuilder
  std::vector<io::DatasetSpecEntry>& vec = this->dataset_spec_.cc_dataset_entries;
  DsetSpecListBuilder registrar(vec, field_info, out_name_recipe, force_buf_choice);

  // Part 2: record the cell-centered fields that will be recorded
  // =============================================================
  // TODO: logic for determining recorded fields should be independent of file format
  if (this->file_format_ == FileFormat::H5_F32) {
    for (int field_id : field_info.get_id_range(field::Kind::HYDRO)) {  // (includes GasEnergy, if applicable)
      std::optional<std::string> maybe_field_name = field_info.field_name(field_id);
      std::string field_name                      = get_or_abort(maybe_field_name);
      std::string param_name                      = "out_float32_" + field_name;
      if (pmap.value_or(param_name, 0)) {
        registrar.add_entry(field_name.c_str(), WriteCond::ALWAYS);
      }
    }

    for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
      std::optional<std::string> maybe_field_name = field_info.field_name(field_id);
      std::string field_name                      = get_or_abort(maybe_field_name);
      std::string param_name                      = "out_float32_" + field_name;
      if (pmap.value_or(param_name, 0)) {
        registrar.add_entry(field_name.c_str(), WriteCond::ALWAYS);
      }
    }

  } else {  // all output formats other than FileFormat::H5_F32
    registrar.add_entry("density", WriteCond::ALWAYS);
    registrar.add_entry("momentum_x", MOMENTUM_CONDITION);
    registrar.add_entry("momentum_y", MOMENTUM_CONDITION);
    registrar.add_entry("momentum_z", MOMENTUM_CONDITION);
    registrar.add_entry("Energy", ENERGY_CONDITION);
#ifdef DE
    registrar.add_entry("GasEnergy", ENERGY_CONDITION);
#endif

    for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
      std::optional<std::string> maybe_field_name = field_info.field_name(field_id);
      std::string name                            = get_or_abort(maybe_field_name);
      if (name == "e_density") {
        registrar.add_entry(name.c_str(), ELECTRONS_CONDITION);
      } else if (name == "metal_density") {
        registrar.add_entry(name.c_str(), METALS_CONDITION);
      } else {
        registrar.add_entry(name.c_str(), WriteCond::ALWAYS);
      }
    }
  }

  // Part 3: record the face-centered fields that will be recorded
  // =============================================================

  // by default we don't expect to write any bfields
  dataset_spec_.write_mag = {false, false, false};

  // this loop is empty if not compiled with MHD
  for (int field_id : field_info.get_id_range(field::Kind::MAGNETIC)) {
    std::optional<std::string> maybe_field_name = field_info.field_name(field_id);
    std::string field_name                      = get_or_abort(maybe_field_name);
    // TODO: logic for determining recorded fields should be independent of file format
    bool write;
    if (this->file_format_ == FileFormat::H5_F32) {
      write = pmap.value_or("out_float32_" + field_name, 0) != 0;
    } else {
      write = true;
    }
    this->dataset_spec_.write_mag[bfield_name_to_012_(field_name)] = write;
  }

  // For now, we don't bothered migrating temperature and gravitational potential
  // away from ifdefs, the handling remains very manual
}

// the following anonymous namespace encloses functions used to help us implement
// FieldWriter::operator()
namespace
{  // stuff inside an anonymous namespace is local to this file

/*! does the heavy-lifting of writing fields to hdf5 files.
 *
 *  \todo
 *  The initial implementation goes out of its way to retain all of the historical
 *  distinctions that occur between regular outputs and float32 outputs. We should
 *  really cut down on these differences!
 */
template <bool ForceF32Output>
void Write_Fields_to_HDF5_helper_(const std::string& filename, Grid3D& G, const io::DatasetSpec& dataset_spec,
                                  io::LazyScratchBuf& lazy_scratch_buf)
{
#ifndef HDF5
  CHOLLA_ERROR(
      "This error should never show up. Somehow FieldWriter was configured "
      "to write hdf5 files when cholla wasn't compiled with hdf5");
#else
  // get the value-type of the dataset buffers
  using T = std::conditional_t<ForceF32Output, float, Real>;

  const Header& H = G.H;
  bool is_3D      = H.nx > 1 and H.ny > 1 and H.nz > 1;

  if (ForceF32Output and not is_3D) {
    // this approximates historical data... Historically we would actually make a file
    // but not record fields to it, but this is close enough!
    return;
  }

  // Create a new file using default properties.
  hid_t file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Allocate necessary buffers
  int nx_dset = H.nx_real;
  int ny_dset = H.ny_real;
  int nz_dset = H.nz_real;
  #ifdef MHD
  size_t buffer_size = (nx_dset + 1) * (ny_dset + 1) * (nz_dset + 1);
  #else
  size_t buffer_size = nx_dset * ny_dset * nz_dset;
  #endif
  //T* dev_dataset_buf  = lazy_scratch_buf.get_buf_dev<T>(buffer_size);
  //T* host_dataset_buf = lazy_scratch_buf.get_buf_host<T>(buffer_size);
  Real* dev_dataset_buf  = lazy_scratch_buf.get_buf_dev<T>(buffer_size);
  Real* host_dataset_buf = lazy_scratch_buf.get_buf_host<T>(buffer_size);

  // write out regular cell-centered fields
  for (const io::DatasetSpecEntry& cur_spec : dataset_spec.cc_dataset_entries) {
    if constexpr (ForceF32Output) {
      // todo: consider more robust behavior here
      CHOLLA_ASSERT(cur_spec.condition == io::WriteCond::ALWAYS, "unexpected case");
      CHOLLA_ASSERT(cur_spec.io_buf == field::IOBuf::DEVICE, "unexpected case");
      Real* ptr = &G.C.device[cur_spec.field_id * H.n_cells];
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, host_dataset_buf, dev_dataset_buf,
                          ptr, cur_spec.name.c_str());
    } else {
      if (cur_spec.condition == io::WriteCond::REQUIRE_COMPLETE_DATA && not H.Output_Complete_Data) continue;
      if (cur_spec.io_buf == field::IOBuf::HOST) {
        Real* ptr = &G.C.host[cur_spec.field_id * H.n_cells];
        Write_Grid_HDF5_Field_CPU(H, file_id, host_dataset_buf, ptr, cur_spec.name.c_str());
      } else {
        Real* ptr = &G.C.device[cur_spec.field_id * H.n_cells];
        Write_Grid_HDF5_Field_GPU(H, file_id, host_dataset_buf, dev_dataset_buf, ptr, cur_spec.name.c_str());
      }
    }
  }

  // write out magnetic fields
  // -> we maintain historical behavior and only do this for 3D simulations
  if (is_3D) {
    const char* dset_names[3] = {"/magnetic_x", "/magnetic_y", "/magnetic_z"};
    for (int i = 0; i < 3; i++) {
      if (not dataset_spec.write_mag[i]) continue;
      const char* field_name            = dset_names[i] + 1;
      std::optional<int> maybe_field_id = G.field_info.field_id(field_name);
      Real* ptr                         = &G.C.device[H.n_cells * get_or_abort(maybe_field_id)];
      if constexpr (ForceF32Output) {
        // TODO (by Alwin, for anyone) : Repair output format if needed and remove the chprintf when appropriate
        chprintf("WARNING: MHD float-32 output has a different output format than float-64\n");
        Write_HDF5_Field_3D(H.nx, H.ny, nx_dset + 1, ny_dset + 1, nz_dset + 1, H.n_ghost - 1, file_id, host_dataset_buf,
                            dev_dataset_buf, ptr, dset_names[i]);
      } else {
        if (not H.Output_Complete_Data) continue;
        int real_shape[3] = {H.nx_real + (i == 0), H.ny_real + (i == 1), H.nz_real + (i == 2)};
        Write_HDF5_Field_3D(H.nx, H.ny, real_shape[0], real_shape[1], real_shape[2], H.n_ghost, file_id,
                            host_dataset_buf, dev_dataset_buf, ptr, dset_names[i], i);
      }
    }
  }

  // handle all of the weird special cases
  if constexpr (not ForceF32Output) {
  #if defined(OUTPUT_TEMPERATURE) && defined(CHEMISTRY_GPU)
    Compute_Gas_Temperature(G.Chem.Fields.temperature_h, false);
    Write_Grid_HDF5_Field_CPU(H, file_id, host_dataset_buf, G.Chem.Fields.temperature_h, "/temperature");
  #elif defined(OUTPUT_TEMPERATURE) && defined(COOLING_GRACKLE)
    Write_Grid_HDF5_Field_CPU(H, file_id, host_dataset_buf, G.Cool.temperature, "/temperature");
  #endif

  #if defined(GRAVITY) && defined(OUTPUT_POTENTIAL)
    if (is_3D) {  // 3D case
      const Grav3D& Grav = G.Grav;
      Write_Generic_HDF5_Field_GPU(Grav.nx_local + 2 * N_GHOST_POTENTIAL, Grav.ny_local + 2 * N_GHOST_POTENTIAL,
                                   Grav.nz_local + 2 * N_GHOST_POTENTIAL, Grav.nx_local, Grav.ny_local, Grav.nz_local,
                                   N_GHOST_POTENTIAL, file_id, host_dataset_buf, dev_dataset_buf, Grav.F.potential_d,
                                   "/grav_potential");
    }
  #endif  // GRAVITY and OUTPUT_POTENTIAL
  }


  // Likely needs editing, check device or host array and offset

  #if defined(RT) && defined(OUTPUT_RADIATION)
    const Rad3D& Rad = G.Rad;

    // add the rsource field
    //Write_Generic_HDF5_Field_GPU(H.nx_real + 2 * H.n_ghost, H.ny_real + 2 * H.n_ghost,
    //                             H.nz_real + 2 * H.n_ghost, H.nx_real, H.ny_real, H.nz_real,
    //                             H.n_ghost, file_id, host_dataset_buf, dev_dataset_buf, Rad.rtFields.dev_rs,
    //                             "/source");
    Real* rtptr = Rad.rtFields.dev_rs;
    Write_Grid_HDF5_Field_GPU(H, file_id, host_dataset_buf, dev_dataset_buf, rtptr, "/source");

    // loop over the number of radiation fields
#ifdef OTVET
    const char* rt_dset_names[7] = {"/intensity", "/rf_HI_near", "/rf_HeI_near","/rf_HeII_near","/rf_HI_far","/rf_HeI_far","/rf_HeII_far"};
#endif
#ifdef M1
    const char* rt_dset_names[16] = {"/rf_intensity", "/rf_intensity_Mx", "/rf_intensity_My","/rf_intensity_Mz",
                                     "/rf_HI",        "/rf_HI_Mx",        "/rf_HI_My",       "/rf_HI_Mz",
                                     "/rf_HeI",       "/rf_HeI_Mx",       "/rf_HeI_My",      "/rf_HeI_Mz",
                                     "/rf_HeII",      "/rf_HeII_Mx",      "/rf_HeII_My",     "/rf_HeII_Mz"};
#endif
    for (int n = 0; n < Rad.n_rf; n++) {
      //char dataset[100];
      //char number[10];
      //sprintf(dataset,"/radiation_");
      //sprintf(number,"%d",n);
      //strcat(dataset, number);
      //Write_Generic_HDF5_Field_GPU(H.nx_real + 2 * H.n_ghost, H.ny_real + 2 * H.n_ghost,
      //                             H.nz_real + 2 * H.n_ghost, H.nx_real, H.ny_real, H.nz_real,
      //                             H.n_ghost, file_id, host_dataset_buf, dev_dataset_buf, &(Rad.rtFields.dev_rf[n * H.n_cells]),
      //                             dataset);
      rtptr = &Rad.rtFields.dev_rf[n * H.n_cells];
      Write_Grid_HDF5_Field_GPU(H, file_id, host_dataset_buf, dev_dataset_buf, rtptr, rt_dset_names[n]);
    }
  #endif  // RT and OUTPUT_RADIATION


  // close the file
  if (H5Fclose(file_id) < 0) {
    CHOLLA_ERROR("File write failed.");
  }
#endif
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
int Record_Colnames_And_Get_Field_Ptrs_(const Real** ptr_arr, bool* is_cell_centered, std::FILE* fp,
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
  const char* field_names[3] = {"magnetic_x", "magnetic_y", "magnetic_z"};
  for (int i = 0; i < 3; i++) {
    if (not dataset_spec.write_mag[i]) continue;

    const char* field_name = field_names[i];
    std::fprintf(fp, "\t%s", field_name);  // <- write the column name

    std::optional<int> maybe_field_id   = field_info.field_id(field_name);
    const Real* ptr                     = &G.C.host[H.n_cells * get_or_abort(maybe_field_id)];
    ptr_arr[field_ptr_counter]          = ptr;
    is_cell_centered[field_ptr_counter] = false;
    field_ptr_counter++;
  }

  std::fputc('\n', fp);

  return field_ptr_counter;
}

/*! Helper function that write the conserved quantities to a text output file
 *
 *  \param[in] filename output file name
 *  \param[in] G specifies all grid data
 *  \param[in] dataset_spec Specifies properties about all fields that may be written
 *      by the current simulation
 *
 *  \note
 *  The fact that the data is interleaved (and the fact that the number of characters
 *  per row is a variable), makes this a little tricky.
 */
void Write_Grid_Text_(const std::string& filename, const Grid3D& G, const DatasetSpec& dataset_spec)
{
  const Header& H             = G.H;
  const Grid3D::Conserved& C  = G.C;
  const FieldInfo& field_info = G.field_info;

  if (H.nx * H.ny * H.nz > 1000) std::printf("Ascii outputs only recommended for small problems!\n");

  // sanity check: the factory method should prevent construction of FieldWriter in
  // circumstances where the following assertion would fail
  bool is_1D = (H.nx > 1 && H.ny == 1 && H.nz == 1);
  CHOLLA_ASSERT(is_1D, "can only write Fields to text files for 1D datasets");

  constexpr int MAX_FIELDS = 20;  // <- make this bigger, if necessary
  {                               // perform some sanity checks!
    const int n_fields = field_info.n_fields();
    CHOLLA_ASSERT(n_fields > 0, "n_names must be positive");
    CHOLLA_ASSERT(n_fields <= MAX_FIELDS, "n_fields %d exceeds MAX_FIELDS, %d", n_fields, MAX_FIELDS);
  }

  // Part 1: Open the file for txt writes and write the header
  // ---------------------------------------------------------
  std::FILE* fp = std::fopen(filename.data(), "w");
  if (fp == nullptr) {
    CHOLLA_ERROR("Error opening output file.");
  }

  // write the header to the output file
  G.Write_Header_Text(fp);

  // Part 2: collect info about each field & write the initial header for the text file
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

  // Part 3: Record all data
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

  // Part 4: Close the output file
  // -----------------------------
  std::fclose(fp);
}

}  // anonymous namespace

void FieldWriter::operator()(Grid3D& G, Parameters P, int nfile, const FnameTemplate& fname_template) const
{
  const char* pre_extension_suffix = (this->file_format_ == FileFormat::H5_F32) ? ".float32" : "";
  std::string filename             = fname_template.format_fname(nfile, pre_extension_suffix);

  switch (this->file_format_) {
    case FileFormat::H5_F32:
      //Write_Fields_to_HDF5_helper_<true>(filename, G, this->dataset_spec_, *this->lazy_scratch_buf_);
      Write_Fields_to_HDF5_helper_<false>(filename, G, this->dataset_spec_, *this->lazy_scratch_buf_);
      return;
    case FileFormat::H5_NATIVE_PRECISION:
      Write_Fields_to_HDF5_helper_<false>(filename, G, this->dataset_spec_, *this->lazy_scratch_buf_);
      return;
    case FileFormat::TEXT:
      Write_Grid_Text_(filename, G, this->dataset_spec_);
      return;
  }
  CHOLLA_ERROR("can't handle specified file format");
}

}  // namespace io
