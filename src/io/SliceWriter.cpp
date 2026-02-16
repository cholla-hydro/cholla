/*!
 * \file
 * Implements the SliceWriter type
 */

#include "SliceWriter.h"

#include <hdf5.h>

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

SliceWriter::SliceWriter(ParameterMap &pmap, const FieldInfo &field_info) {}

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

// this isn't very useful (yet)
struct SliceProps {
  PlaneChoice choice;
  const char *name_suffix;
  hid_t dataspace_id;
};

}  // anonymous namespace

#ifdef HDF5
/*! Helper function that does most heavy lifting for writing HDF5 slices */
static void Write_Slices_HDF5_(const Grid3D &G, hid_t file_id)
{
  const Header &H = G.H;
  bool is_3D      = H.nx > 1 && H.ny > 1 && H.nz > 1;
  if (not is_3D) {
    chprintf("Slice write only works for 3D data.\n");
    return;
  }
  const FieldInfo &field_info = G.field_info;
  const Grid3D::Conserved &C  = G.C;

  const std::map<std::string, std::string> name_map{{"density", "d"},     {"momentum_x", "mx"}, {"momentum_y", "my"},
                                                    {"momentum_z", "mz"}, {"Energy", "E"},      {"GasEnergy", "GE"}};

  int i, j, k, id, buf_id;
  #ifdef SCALAR
  Real *dataset_buffer_scalar;
  #endif
  herr_t status;

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

    for (int field_id : field_info.get_id_range(field::Kind::HYDRO)) {
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

      std::string field_name = field_info.field_name(field_id).value();
      std::string dset_name  = '/' + name_map.at(field_name) + slice_prop.name_suffix;

      status = Write_HDF5_Dataset(file_id, slice_prop.dataspace_id, buf.data(), dset_name.c_str());
    }
  }

  // Allocate memory for the xy slices
  #ifdef MHD
  std::vector<Real> dataset_buffer_magnetic_x(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_magnetic_y(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_magnetic_z(H.nx_real * H.ny_real);
  #endif  // MHD
  #ifdef SCALAR
  dataset_buffer_scalar = (Real *)malloc(NSCALARS * H.nx_real * H.ny_real * sizeof(Real));
  #endif

  // Copy the xy slices to the memory buffers
  for (j = 0; j < H.ny_real; j++) {
    for (i = 0; i < H.nx_real; i++) {
      buf_id     = j + i * H.ny_real;
      int zslice = global_slice_idx.z();
      // check whether the slice intersects the current process's local domain
      if (zslice >= idx_local_start.z() && zslice < idx_local_start.z() + nz_local) {
        id = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice - idx_local_start.z() + H.n_ghost,
                                            H.nx, H.ny);
  #ifdef MHD
        int id_xm1                        = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost,
                                                                           zslice - idx_local_start.z() + H.n_ghost, H.nx, H.ny);
        int id_ym1                        = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1,
                                                                           zslice - idx_local_start.z() + H.n_ghost, H.nx, H.ny);
        int id_zm1                        = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost,
                                                                           zslice - idx_local_start.z() + H.n_ghost - 1, H.nx, H.ny);
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.ny] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.ny] = 0;
        }
  #endif
      }
    }
  }

  // Write out the xy datasets for each variable
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xy");
  #endif  // MHD
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_scalar, "/scalar_xy");
  #endif

  // free the dataset buffers
  #ifdef SCALAR
  free(dataset_buffer_scalar);
  #endif

  // allocate the memory for the xz slices
  #ifdef MHD
  dataset_buffer_magnetic_x.resize(H.nx_real * H.nz_real);
  dataset_buffer_magnetic_y.resize(H.nx_real * H.nz_real);
  dataset_buffer_magnetic_z.resize(H.nx_real * H.nz_real);
  #endif  // MHD
  #ifdef SCALAR
  dataset_buffer_scalar = (Real *)malloc(NSCALARS * H.nx_real * H.nz_real * sizeof(Real));
  #endif

  // Copy the xz slices to the memory buffers
  for (k = 0; k < H.nz_real; k++) {
    for (i = 0; i < H.nx_real; i++) {
      buf_id     = k + i * H.nz_real;
      int yslice = global_slice_idx.y();
      // check whether the slice intersects the current process's local domain
      if (yslice >= idx_local_start.y() && yslice < idx_local_start.y() + ny_local) {
        id = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - idx_local_start.y() + H.n_ghost, k + H.n_ghost,
                                            H.nx, H.ny);
  #ifdef MHD
        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, yslice - idx_local_start.y() + H.n_ghost,
                                                    k + H.n_ghost, H.nx, H.ny);
        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - idx_local_start.y() + H.n_ghost - 1,
                                                    k + H.n_ghost, H.nx, H.ny);
        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - idx_local_start.y() + H.n_ghost,
                                                    k + H.n_ghost - 1, H.nx, H.ny);
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.nz] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.nz] = 0;
        }
  #endif
      }
    }
  }

  // Write out the xz datasets for each variable
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xz");
  #endif  // MHD
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_scalar, "/scalar_xz");
  #endif

  // free the dataset buffers
  #ifdef SCALAR
  free(dataset_buffer_scalar);
  #endif

  // allocate the memory for the yz slices
  #ifdef MHD
  dataset_buffer_magnetic_x.resize(H.ny_real * H.nz_real);
  dataset_buffer_magnetic_y.resize(H.ny_real * H.nz_real);
  dataset_buffer_magnetic_z.resize(H.ny_real * H.nz_real);
  #endif  // MHD
  #ifdef SCALAR
  dataset_buffer_scalar = (Real *)malloc(NSCALARS * H.ny_real * H.nz_real * sizeof(Real));
  #endif

  // Copy the yz slices to the memory buffers
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      buf_id     = k + j * H.nz_real;
      int xslice = global_slice_idx.x();
      // check whether the slice intersects the current process's local domain
      if (xslice >= idx_local_start.x() && xslice < idx_local_start.x() + nx_local) {
        id = cuda_utilities::compute1DIndex(xslice - idx_local_start.x(), j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
  #ifdef MHD
        int id_xm1 =
            cuda_utilities::compute1DIndex(xslice - idx_local_start.x() - 1, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
        int id_ym1 =
            cuda_utilities::compute1DIndex(xslice - idx_local_start.x(), j + H.n_ghost - 1, k + H.n_ghost, H.nx, H.ny);
        int id_zm1 =
            cuda_utilities::compute1DIndex(xslice - idx_local_start.x(), j + H.n_ghost, k + H.n_ghost - 1, H.nx, H.ny);
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.ny * H.nz] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.ny * H.nz] = 0;
        }
  #endif
      }
    }
  }

  // Write out the yz datasets for each variable
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_yz");
  #endif  // MHD
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_scalar, "/scalar_yz");
  #endif

  // free the dataset buffers
  #ifdef SCALAR
  free(dataset_buffer_scalar);
  #endif

  // free the dataspace ids
  for (const SliceProps &slice_prop : slice_props) {
    status = H5Sclose(slice_prop.dataspace_id);
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
  Write_Slices_HDF5_(G, file_id);

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