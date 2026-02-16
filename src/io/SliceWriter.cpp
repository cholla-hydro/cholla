/*!
 * \file
 * Implements the SliceWriter type
 */

#include "SliceWriter.h"

#ifndef HDF5
#include <hdf5.h>
#endif

#include <algorithm>  // std::fill, std::max
#include <cstddef>    // ptrdiff_t
#include <map>
#include <string>

#include "../grid/field_info.h"
#include "../io/FnameTemplate.h"
#include "../io/io.h"
#include "../utils/basic_structs.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"

namespace io
{

SliceWriter::SliceWriter(ParameterMap &pmap, const FieldInfo &field_info)
{
  // this maps field names to shorttened dataset names
  // -> this is for historical purposes. Honestly, I think it would be better to
  //    preserve the full field name
  const std::map<std::string, std::string> name_map{{"density", "d"},     {"momentum_x", "mx"}, {"momentum_y", "my"},
                                                    {"momentum_z", "mz"}, {"Energy", "E"},      {"GasEnergy", "GE"}};
  // append entries to cc_field_id_dset_name_pairs_ for each cell-centered hydro field
  for (int field_id : field_info.get_id_range(field::Kind::HYDRO)) {
    std::string field_name = field_info.field_name(field_id).value();
    std::string dset_name;
    auto search = name_map.find(field_name);
    if (search == name_map.end()) {
      // in this case, I REALLY think we should stick with the regular field name
      CHOLLA_ERROR("there isn't a standard short-name for the \"%s\" field name", field_name.c_str());
    } else {
      dset_name = search->second;
    }
    this->cc_field_id_dset_name_pairs_.emplace_back(field_id, dset_name);
  }

  // if there are any passive scalars, we will ONLY save the very first one in a dataset
  // named scalar
  // -> frankly, I think we would be better off just ignoring this historical convention
  //    and saving slices of all passive scalars to datasets that use the field's name
  for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
    this->cc_field_id_dset_name_pairs_.emplace_back(field_id, "scalar");
  }
}

namespace
{  // stuff inside an anonymous namespace is local to this file

enum struct PlaneChoice { XY, XZ, YZ };

/*! Fill the output buffer, \p, for a slice.
 *
 *  To describe this function, we make use of the term "active-zone index". This refers
 *  to a location in the local "active-zone" (i.e. it is the same as the regular index
 *  when the number of ghost zones is 0)
 *
 *  \param[out] buf The output buffer
 *  \param[in] local_active_zone_shape Specifies the shape of the local active zone
 *  \param[in] choice Specifies the kind of slice to make
 *  \param[in] local_active_zone_slice_idx The cell-centered "active-zone index" that
 *      the slice is made along. The precise interpretation depends on \p choice. This
 *      is along the z-axis for \ref PlaneChoice::XY, along the y-axis for
 *      \ref PlaneChoice::XZ, and along the x-index for \ref PlaneChoice::YZ.
 *  \param[in] f The callback function that gets called as `f(xid, yid, zid)` and
 *      returns the corresponding cell-centered value (as a `Real`). Importantly, `xid`,
 *      `yid`, and `zid` specify the "active-zone indices".
 */
template <typename Fn>
void Fill_Slice_Buf_(Real *buf, const hydro_utilities::VectorXYZ<int> &local_active_zone_shape, PlaneChoice choice,
                     int local_active_zone_slice_idx, Fn f)
{
  int nz_real = local_active_zone_shape.z();
  int ny_real = local_active_zone_shape.y();
  int nx_real = local_active_zone_shape.x();
  switch (choice) {
    case PlaneChoice::XY: {
      int k = local_active_zone_slice_idx;
      for (int j = 0; j < ny_real; j++) {
        for (int i = 0; i < nx_real; i++) {
          buf[j + i * ny_real] = f(i, j, k);
        }
      }
      return;
    }
    case PlaneChoice::XZ: {
      int j = local_active_zone_slice_idx;
      for (int k = 0; k < nz_real; k++) {
        for (int i = 0; i < nx_real; i++) {
          buf[k + i * nz_real] = f(i, j, k);
        }
      }
      return;
    }
    case PlaneChoice::YZ: {
      int i = local_active_zone_slice_idx;
      for (int k = 0; k < nz_real; k++) {
        for (int j = 0; j < ny_real; j++) {
          buf[k + j * nz_real] = f(i, j, k);
        }
      }
      return;
    }
  }
  CHOLLA_ERROR("Received unknown PlaneChoice");
}

}  // anonymous namespace

#ifdef HDF5

// this isn't very useful (yet)
struct SliceProps {
  PlaneChoice choice;
  const char *name_suffix;
  hid_t dataspace_id;
};

/*! Helper function that does most heavy lifting for writing HDF5 slices */
static void Write_Slices_HDF5_(const Grid3D &G, hid_t file_id,
                               const std::vector<std::pair<int, std::string>> &cc_field_id_dset_name_pairs)
{
  const Header &H = G.H;
  bool is_3D      = H.nx > 1 && H.ny > 1 && H.nz > 1;
  if (not is_3D) {
    chprintf("Slice write only works for 3D data.\n");
    return;
  }
  const FieldInfo &field_info = G.field_info;
  const Grid3D::Conserved &C  = G.C;
  const bool using_MHD        = field_info.n_fields(field::Kind::MAGNETIC) > 0;

  #ifndef MPI_CHOLLA
  const hydro_utilities::VectorXYZ<ptrdiff_t> global_slice_idx{H.nx / 2, H.ny / 2, H.nz / 2};
  const hydro_utilities::VectorXYZ<ptrdiff_t> idx_local_start{0, 0, 0};
  #else  // defined(MPI_CHOLLA)
  const hydro_utilities::VectorXYZ<ptrdiff_t> global_slice_idx{nx_global / 2, ny_global / 2, nz_global / 2};
  const hydro_utilities::VectorXYZ<ptrdiff_t> idx_local_start{nx_local_start, ny_local_start, nz_local_start};
  #endif

  const hydro_utilities::VectorXYZ<int> local_active_zone_shape{H.nx_real, H.ny_real, H.nz_real};

  // Create the slice_props
  hsize_t dims_xy[2] = {static_cast<hsize_t>(H.nx_real), static_cast<hsize_t>(H.ny_real)};
  hsize_t dims_xz[2] = {static_cast<hsize_t>(H.nx_real), static_cast<hsize_t>(H.nz_real)};
  hsize_t dims_yz[2] = {static_cast<hsize_t>(H.ny_real), static_cast<hsize_t>(H.nz_real)};

  SliceProps slice_props[3] = {{PlaneChoice::XY, "_xy", H5Screate_simple(2, dims_xy, nullptr)},
                               {PlaneChoice::XZ, "_xz", H5Screate_simple(2, dims_xz, nullptr)},
                               {PlaneChoice::YZ, "_yz", H5Screate_simple(2, dims_yz, nullptr)}};

  const int n_ghost = H.n_ghost;
  const int nx      = H.nx;
  const int ny      = H.ny;

  int buf_len = std::max({H.nx_real * H.ny_real, H.nx_real * H.ny_real, H.ny_real * H.nz_real});
  std::vector<Real> buf(buf_len);

  for (const SliceProps &slice_prop : slice_props) {
    int local_active_zone_slice_idx;
    bool slice_intersects_local_domain;
    switch (slice_prop.choice) {
      case PlaneChoice::XY:
        local_active_zone_slice_idx   = global_slice_idx.z() - idx_local_start.z();
        slice_intersects_local_domain = (local_active_zone_slice_idx >= 0) and (local_active_zone_slice_idx < nz_local);
        break;
      case PlaneChoice::XZ:
        local_active_zone_slice_idx   = global_slice_idx.y() - idx_local_start.y();
        slice_intersects_local_domain = (local_active_zone_slice_idx >= 0) and (local_active_zone_slice_idx < ny_local);
        break;
      case PlaneChoice::YZ:
        local_active_zone_slice_idx   = global_slice_idx.x() - idx_local_start.x();
        slice_intersects_local_domain = (local_active_zone_slice_idx >= 0) and (local_active_zone_slice_idx < nx_local);
        break;
      default:
        CHOLLA_ERROR("Unrecognized PlaneChoice");
    }

    // record slices of all cell-centered fields
    for (const std::pair<int, std::string> pair : cc_field_id_dset_name_pairs) {
      int field_id          = pair.first;
      std::string dset_name = pair.second;

      const Real *ptr = &C.host[field_id * H.n_cells];
      auto get_val    = [=](int active_zone_xid, int active_zone_yid, int active_zone_zid) -> Real {
        int xid = active_zone_xid + n_ghost;
        int yid = active_zone_yid + n_ghost;
        int zid = active_zone_zid + n_ghost;
        return ptr[cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny)];
      };

      // check whether the slice intersects the current process's local domain
      if (slice_intersects_local_domain) {
        Fill_Slice_Buf_(buf.data(), local_active_zone_shape, slice_prop.choice, local_active_zone_slice_idx, get_val);
      } else {
        std::fill(buf.begin(), buf.end(), static_cast<Real>(0.0));
      }

      std::string full_dset_name = '/' + dset_name + slice_prop.name_suffix;

      herr_t status = Write_HDF5_Dataset(file_id, slice_prop.dataspace_id, buf.data(), full_dset_name.c_str());
    }

    for (int field_id : field_info.get_id_range(field::Kind::MAGNETIC)) {
      std::string field_name = field_info.field_name(field_id).value();

      if (slice_intersects_local_domain) {
        char comp       = field_name[field_name.size() - 1];  // <- holds 'x', 'y', or 'z'
        int field_id    = field_info.field_id(field_name).value();
        const Real *ptr = &C.host[field_id * H.n_cells];
        const hydro_utilities::VectorXYZ<int> off_L{n_ghost - (comp == 'x'), n_ghost - (comp == 'y'),
                                                    n_ghost - (comp == 'z')};
        const hydro_utilities::VectorXYZ<int> off_R{n_ghost, n_ghost, n_ghost};

        auto get_val = [=](int active_zone_xid, int active_zone_yid, int active_zone_zid) -> Real {
          int i_L = cuda_utilities::compute1DIndex(active_zone_xid + off_L.x(), active_zone_yid + off_L.y(),
                                                   active_zone_zid + off_L.z(), nx, ny);
          int i_R = cuda_utilities::compute1DIndex(active_zone_xid + off_R.x(), active_zone_yid + off_R.y(),
                                                   active_zone_zid + off_R.z(), nx, ny);
          return 0.5 * (ptr[i_L] + ptr[i_R]);
        };

        Fill_Slice_Buf_(buf.data(), local_active_zone_shape, slice_prop.choice, local_active_zone_slice_idx, get_val);
      } else {
        std::fill(buf.begin(), buf.end(), static_cast<Real>(0.0));
      }

      std::string full_dset_name = '/' + field_name + slice_prop.name_suffix;

      herr_t status = Write_HDF5_Dataset(file_id, slice_prop.dataspace_id, buf.data(), full_dset_name.c_str());
    }
  }

  // free the dataspace ids
  for (const SliceProps &slice_prop : slice_props) {
    herr_t status = H5Sclose(slice_prop.dataspace_id);
  }
}
#endif  // HDF5

void SliceWriter::operator()(Grid3D &G, struct Parameters P, int nfile, const FnameTemplate &fname_template) const
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename = fname_template.format_fname(nfile, "_slice");

  // Create a new file
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write slices of all variables to the output file
  Write_Slices_HDF5_(G, file_id, cc_field_id_dset_name_pairs_);

  // Close the file
  status = H5Fclose(file_id);

  #ifdef MPI_CHOLLA
  if (status < 0) {
    printf("Output_Slices: File write failed. ProcID: %d\n", procID);
    chexit(-1);
  }
  #else   // MPI_CHOLLA is not defined
  if (status < 0) {
    printf("Output_Slices: File write failed.\n");
    exit(-1);
  }
  #endif  // MPI_CHOLLA
#else     // HDF5 is not defined
  printf("Output_Slices only defined for hdf5 writes.\n");
#endif    // HDF5
}

}  // namespace io