/*!
 * \file
 * Implements the SliceWriter type
 */

#include "SliceWriter.h"

#include <hdf5.h>

#include <cstddef>  // ptrdiff_t

#include "../io/FnameTemplate.h"
#include "../io/io.h"
#include "../utils/basic_structs.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"

namespace io
{

SliceWriter::SliceWriter(ParameterMap &pmap, const FieldInfo &field_info) {}

// this isn't very useful (yet)
struct SliceProps {
  const char *name_suffix;
  hid_t dataspace_id;
};

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

  const Grid3D::Conserved &C = G.C;
  int i, j, k, id, buf_id;
  Real *dataset_buffer_d;
  Real *dataset_buffer_mx;
  Real *dataset_buffer_my;
  Real *dataset_buffer_mz;
  Real *dataset_buffer_E;
  #ifdef DE
  Real *dataset_buffer_GE;
  #endif
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

  // Create the data spaces for the datasets
  hsize_t dims_xy[2] = {static_cast<hsize_t>(H.nx_real), static_cast<hsize_t>(H.ny_real)};
  hsize_t dims_xz[2] = {static_cast<hsize_t>(H.nx_real), static_cast<hsize_t>(H.nz_real)};
  hsize_t dims_yz[2] = {static_cast<hsize_t>(H.ny_real), static_cast<hsize_t>(H.nz_real)};

  SliceProps slice_props[3] = {{"xy", H5Screate_simple(2, dims_xy, nullptr)},
                               {"xz", H5Screate_simple(2, dims_xz, nullptr)},
                               {"yz", H5Screate_simple(2, dims_yz, nullptr)}};

  // Allocate memory for the xy slices
  dataset_buffer_d  = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  dataset_buffer_mx = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  dataset_buffer_my = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  dataset_buffer_mz = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  dataset_buffer_E  = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  #ifdef MHD
  std::vector<Real> dataset_buffer_magnetic_x(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_magnetic_y(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_magnetic_z(H.nx_real * H.ny_real);
  #endif  // MHD
  #ifdef DE
  dataset_buffer_GE = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
  #endif
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
        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost,
                                                    zslice - idx_local_start.z() + H.n_ghost, H.nx, H.ny);
        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1,
                                                    zslice - idx_local_start.z() + H.n_ghost, H.nx, H.ny);
        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost,
                                                    zslice - idx_local_start.z() + H.n_ghost - 1, H.nx, H.ny);
  #endif  // MHD
        dataset_buffer_d[buf_id]  = C.density[id];
        dataset_buffer_mx[buf_id] = C.momentum_x[id];
        dataset_buffer_my[buf_id] = C.momentum_y[id];
        dataset_buffer_mz[buf_id] = C.momentum_z[id];
        dataset_buffer_E[buf_id]  = C.Energy[id];
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = C.GasEnergy[id];
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.ny] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
        dataset_buffer_d[buf_id]  = 0;
        dataset_buffer_mx[buf_id] = 0;
        dataset_buffer_my[buf_id] = 0;
        dataset_buffer_mz[buf_id] = 0;
        dataset_buffer_E[buf_id]  = 0;
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = 0;
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.ny] = 0;
        }
  #endif
      }
    }
  }

  // Write out the xy datasets for each variable
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_d, "/d_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_mx, "/mx_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_my, "/my_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_mz, "/mz_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_E, "/E_xy");
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xy");
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xy");
  #endif  // MHD
  #ifdef DE
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_GE, "/GE_xy");
  #endif
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[0].dataspace_id, dataset_buffer_scalar, "/scalar_xy");
  #endif

  // free the dataset buffers
  free(dataset_buffer_d);
  free(dataset_buffer_mx);
  free(dataset_buffer_my);
  free(dataset_buffer_mz);
  free(dataset_buffer_E);
  #ifdef DE
  free(dataset_buffer_GE);
  #endif
  #ifdef SCALAR
  free(dataset_buffer_scalar);
  #endif

  // allocate the memory for the xz slices
  dataset_buffer_d  = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  dataset_buffer_mx = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  dataset_buffer_my = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  dataset_buffer_mz = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  dataset_buffer_E  = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  #ifdef MHD
  dataset_buffer_magnetic_x.resize(H.nx_real * H.nz_real);
  dataset_buffer_magnetic_y.resize(H.nx_real * H.nz_real);
  dataset_buffer_magnetic_z.resize(H.nx_real * H.nz_real);
  #endif  // MHD
  #ifdef DE
  dataset_buffer_GE = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  #endif
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
  #endif  // MHD
        dataset_buffer_d[buf_id]  = C.density[id];
        dataset_buffer_mx[buf_id] = C.momentum_x[id];
        dataset_buffer_my[buf_id] = C.momentum_y[id];
        dataset_buffer_mz[buf_id] = C.momentum_z[id];
        dataset_buffer_E[buf_id]  = C.Energy[id];
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = C.GasEnergy[id];
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.nz] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
        dataset_buffer_d[buf_id]  = 0;
        dataset_buffer_mx[buf_id] = 0;
        dataset_buffer_my[buf_id] = 0;
        dataset_buffer_mz[buf_id] = 0;
        dataset_buffer_E[buf_id]  = 0;
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = 0;
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.nx * H.nz] = 0;
        }
  #endif
      }
    }
  }

  // Write out the xz datasets for each variable
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_d, "/d_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_mx, "/mx_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_my, "/my_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_mz, "/mz_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_E, "/E_xz");
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xz");
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xz");
  #endif  // MHD
  #ifdef DE
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_GE, "/GE_xz");
  #endif
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[1].dataspace_id, dataset_buffer_scalar, "/scalar_xz");
  #endif

  // free the dataset buffers
  free(dataset_buffer_d);
  free(dataset_buffer_mx);
  free(dataset_buffer_my);
  free(dataset_buffer_mz);
  free(dataset_buffer_E);
  #ifdef DE
  free(dataset_buffer_GE);
  #endif
  #ifdef SCALAR
  free(dataset_buffer_scalar);
  #endif

  // allocate the memory for the yz slices
  dataset_buffer_d  = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  dataset_buffer_mx = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  dataset_buffer_my = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  dataset_buffer_mz = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  dataset_buffer_E  = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  #ifdef MHD
  dataset_buffer_magnetic_x.resize(H.ny_real * H.nz_real);
  dataset_buffer_magnetic_y.resize(H.ny_real * H.nz_real);
  dataset_buffer_magnetic_z.resize(H.ny_real * H.nz_real);
  #endif  // MHD
  #ifdef DE
  dataset_buffer_GE = (Real *)malloc(H.ny_real * H.nz_real * sizeof(Real));
  #endif
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
  #endif  // MHD
        dataset_buffer_d[buf_id]  = C.density[id];
        dataset_buffer_mx[buf_id] = C.momentum_x[id];
        dataset_buffer_my[buf_id] = C.momentum_y[id];
        dataset_buffer_mz[buf_id] = C.momentum_z[id];
        dataset_buffer_E[buf_id]  = C.Energy[id];
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0.5 * (C.magnetic_x[id] + C.magnetic_x[id_xm1]);
        dataset_buffer_magnetic_y[buf_id] = 0.5 * (C.magnetic_y[id] + C.magnetic_y[id_ym1]);
        dataset_buffer_magnetic_z[buf_id] = 0.5 * (C.magnetic_z[id] + C.magnetic_z[id_zm1]);
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = C.GasEnergy[id];
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.ny * H.nz] = C.scalar[id + ii * H.n_cells];
        }
  #endif
      } else {
        // write zeros if slice doesn't intersect the current process's local domain
        dataset_buffer_d[buf_id]  = 0;
        dataset_buffer_mx[buf_id] = 0;
        dataset_buffer_my[buf_id] = 0;
        dataset_buffer_mz[buf_id] = 0;
        dataset_buffer_E[buf_id]  = 0;
  #ifdef MHD
        dataset_buffer_magnetic_x[buf_id] = 0;
        dataset_buffer_magnetic_y[buf_id] = 0;
        dataset_buffer_magnetic_z[buf_id] = 0;
  #endif  // MHD
  #ifdef DE
        dataset_buffer_GE[buf_id] = 0;
  #endif
  #ifdef SCALAR
        for (int ii = 0; ii < NSCALARS; ii++) {
          dataset_buffer_scalar[buf_id + ii * H.ny * H.nz] = 0;
        }
  #endif
      }
    }
  }

  // Write out the yz datasets for each variable
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_d, "/d_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_mx, "/mx_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_my, "/my_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_mz, "/mz_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_E, "/E_yz");
  #ifdef MHD
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_yz");
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_yz");
  #endif  // MHD
  #ifdef DE
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_GE, "/GE_yz");
  #endif
  #ifdef SCALAR
  // it turns out that due to an oversight, we *only* write the very first scalar
  status = Write_HDF5_Dataset(file_id, slice_props[2].dataspace_id, dataset_buffer_scalar, "/scalar_yz");
  #endif

  // free the dataset buffers
  free(dataset_buffer_d);
  free(dataset_buffer_mx);
  free(dataset_buffer_my);
  free(dataset_buffer_mz);
  free(dataset_buffer_E);
  #ifdef DE
  free(dataset_buffer_GE);
  #endif
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