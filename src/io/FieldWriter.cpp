/*!
 * \file
 * Implements the FieldWriter type
 */

#include "../io/FieldWriter.h"

#include <cstdio>
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
  std::vector<io::DatasetSpecEntry> &vec;
  const FieldInfo &field_info;

  void add_entry(const char *name, WriteCond cond)
  {
    std::optional<int> maybe_field_id = field_info.field_id(name);
    if (!maybe_field_id.has_value()) {
      CHOLLA_ERROR("the current Cholla config has no \"%s\" field", name);
    }
    int field_id = maybe_field_id.value();
    vec.push_back({field_id, '/' + std::string(name), field_info.io_buf(field_id).value(), cond});
  }
};

/*! does the heavy-lifting of writing fields to hdf5 files.
 *
 *  \todo
 *  The initial implementation goes out of its way to retain all of the historical
 *  distinctions that occur between regular outputs and float32 outputs. We should
 *  really cut down on these differences!
 */
template <bool ForceF32Output>
void Write_Fields_to_HDF5_helper_(const std::string &filename, Grid3D &G, const io::DatasetSpec &dataset_spec,
                                  io::LazyScratchBuf &lazy_scratch_buf)
{
#ifdef HDF5
  // get the value-type of the dataset buffers
  using T = std::conditional_t<ForceF32Output, float, Real>;

  const Header &H = G.H;
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
  T *dev_dataset_buf  = lazy_scratch_buf.get_buf_dev<T>(buffer_size);
  T *host_dataset_buf = lazy_scratch_buf.get_buf_host<T>(buffer_size);

  // write out regular cell-centered fields
  for (const io::DatasetSpecEntry &cur_spec : dataset_spec.cc_dataset_entries) {
    if constexpr (ForceF32Output) {
      // todo: consider more robust behavior here
      CHOLLA_ASSERT(cur_spec.condition == io::WriteCond::ALWAYS, "unexpected case");
      CHOLLA_ASSERT(cur_spec.io_buf == field::IOBuf::DEVICE, "unexpected case");
      Real *ptr = &G.C.device[cur_spec.field_id * H.n_cells];
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, host_dataset_buf, dev_dataset_buf,
                          ptr, cur_spec.name.c_str());
    } else {
      if (cur_spec.condition == io::WriteCond::REQUIRE_COMPLETE_DATA && not H.Output_Complete_Data) continue;
      if (cur_spec.io_buf == field::IOBuf::HOST) {
        Real *ptr = &G.C.host[cur_spec.field_id * H.n_cells];
        Write_Grid_HDF5_Field_CPU(H, file_id, host_dataset_buf, ptr, cur_spec.name.c_str());
      } else {
        Real *ptr = &G.C.device[cur_spec.field_id * H.n_cells];
        Write_Grid_HDF5_Field_GPU(H, file_id, host_dataset_buf, dev_dataset_buf, ptr, cur_spec.name.c_str());
      }
    }
  }

  // write out magnetic fields
  // -> we maintain historical behavior and only do this for 3D simulations
  if (is_3D) {
    const char *dset_names[3] = {"/magnetic_x", "/magnetic_y", "/magnetic_z"};
    for (int i = 0; i < 3; i++) {
      if (not dataset_spec.write_mag[i]) continue;
      const char *field_name = dset_names[i] + 1;
      Real *ptr              = &G.C.device[H.n_cells * G.field_info.field_id(field_name).value()];
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
      const Grav3D &Grav = G.Grav;
      Write_Generic_HDF5_Field_GPU(Grav.nx_local + 2 * N_GHOST_POTENTIAL, Grav.ny_local + 2 * N_GHOST_POTENTIAL,
                                   Grav.nz_local + 2 * N_GHOST_POTENTIAL, Grav.nx_local, Grav.ny_local, Grav.nz_local,
                                   N_GHOST_POTENTIAL, file_id, host_dataset_buf, dev_dataset_buf, Grav.F.potential_d,
                                   "/grav_potential");
    }
  #endif  // GRAVITY and OUTPUT_POTENTIAL
  }

  // close the file
  herr_t status = H5Fclose(file_id);

  if (status < 0) {
    printf("File write failed.\n");
    exit(-1);
  }
#endif
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

FieldWriter::FieldWriter(ParameterMap &pmap, const FieldInfo &field_info) : lazy_scratch_buf_(new LazyScratchBuf)
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

  // For now, we don't bothered migrating temperature and gravitational potential
  // away from ifdefs, the handling remains very manual
}

void io::FieldWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const
{
  // create the filename
  std::string filename = fname_template.format_fname(nfile, "");

#ifdef HDF5
  // create the file for hdf5 writes
  Write_Fields_to_HDF5_helper_<false>(filename, G, this->h5_dataset_spec_, *this->lazy_scratch_buf_);

#else
  if (G.H.nx * G.H.ny * G.H.nz > 1000) printf("Ascii outputs only recommended for small problems!\n");

  // open the file for txt writes
  FILE *out;
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

F32FieldWriter::F32FieldWriter(ParameterMap &pmap, const FieldInfo &field_info) : lazy_scratch_buf_(new LazyScratchBuf)
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

  auto map_name_to_idx = [](const std::string &field_name) -> int {
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
    std::string field_name                                     = field_info.field_name(field_id).value();
    bool write_field                                           = pmap.value_or("out_float32_" + field_name, 0) != 0;
    this->dataset_spec_.write_mag[map_name_to_idx(field_name)] = write_field;
  }
}

void io::F32FieldWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const
{
#ifdef HDF5
  std::string filename = fname_template.format_fname(nfile, ".float32");
  Write_Fields_to_HDF5_helper_<true>(filename, G, this->dataset_spec_, *this->lazy_scratch_buf_);
#endif  // HDF5
}

}  // namespace io