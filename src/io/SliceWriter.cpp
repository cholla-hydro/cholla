/*!
 * \file
 * Implements the SliceWriter type
 */

#include "SliceWriter.h"

#include <hdf5.h>

#include "../io/FnameTemplate.h"
#include "../io/io.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"

namespace io
{

SliceWriter::SliceWriter(ParameterMap &pmap, const FieldInfo &field_info) {}

#ifdef HDF5
/*! Helper function that does most heavy lifting for writing HDF5 slices */
static void Write_Slices_HDF5_(const Grid3D &G, hid_t file_id)
{
  const Header &H            = G.H;
  const Grid3D::Conserved &C = G.C;
  int i, j, k, id, buf_id;
  hid_t dataset_id, dataspace_id;
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
  int xslice, yslice, zslice;
  xslice = H.nx / 2;
  yslice = H.ny / 2;
  zslice = H.nz / 2;
  #ifdef MPI_CHOLLA
  xslice = nx_global / 2;
  yslice = ny_global / 2;
  zslice = nz_global / 2;
  #endif

  // 3D
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    int nx_dset = H.nx_real;
    int ny_dset = H.ny_real;
    int nz_dset = H.nz_real;
    hsize_t dims[2];

    // Create the xy data space for the datasets
    dims[0]      = nx_dset;
    dims[1]      = ny_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

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
        id     = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice, H.nx, H.ny);
        buf_id = j + i * H.ny_real;
  #ifdef MHD
        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost, zslice, H.nx, H.ny);
        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1, zslice, H.nx, H.ny);
        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice - 1, H.nx, H.ny);
  #endif  // MHD
  #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in
        // your domain
        if (zslice >= nz_local_start && zslice < nz_local_start + nz_local) {
          id = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost, zslice - nz_local_start + H.n_ghost, H.nx,
                                              H.ny);
    #ifdef MHD
          int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, j + H.n_ghost,
                                                      zslice - nz_local_start + H.n_ghost, H.nx, H.ny);
          int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost - 1,
                                                      zslice - nz_local_start + H.n_ghost, H.nx, H.ny);
          int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, j + H.n_ghost,
                                                      zslice - nz_local_start + H.n_ghost - 1, H.nx, H.ny);
    #endif  // MHD
  #endif    // MPI_CHOLLA
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
  #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
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
  #endif  // MPI_CHOLLA
      }
    }

    // Write out the xy datasets for each variable
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_d, "/d_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mx, "/mx_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_my, "/my_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mz, "/mz_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_E, "/E_xy");
  #ifdef MHD
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xy");
  #endif  // MHD
  #ifdef DE
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_GE, "/GE_xy");
  #endif
  #ifdef SCALAR
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_scalar, "/scalar_xy");
  #endif
    // Free the dataspace id
    status = H5Sclose(dataspace_id);

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

    // Create the xz data space for the datasets
    dims[0]      = nx_dset;
    dims[1]      = nz_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

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
        id     = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice, k + H.n_ghost, H.nx, H.ny);
        buf_id = k + i * H.nz_real;
  #ifdef MHD
        int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, yslice, k + H.n_ghost, H.nx, H.ny);
        int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - 1, k + H.n_ghost, H.nx, H.ny);
        int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice, k + H.n_ghost - 1, H.nx, H.ny);
  #endif  // MHD
  #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in
        // your domain
        if (yslice >= ny_local_start && yslice < ny_local_start + ny_local) {
          id = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost, k + H.n_ghost, H.nx,
                                              H.ny);
    #ifdef MHD
          int id_xm1 = cuda_utilities::compute1DIndex(i + H.n_ghost - 1, yslice - ny_local_start + H.n_ghost,
                                                      k + H.n_ghost, H.nx, H.ny);
          int id_ym1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost - 1,
                                                      k + H.n_ghost, H.nx, H.ny);
          int id_zm1 = cuda_utilities::compute1DIndex(i + H.n_ghost, yslice - ny_local_start + H.n_ghost,
                                                      k + H.n_ghost - 1, H.nx, H.ny);
    #endif  // MHD
  #endif    // MPI_CHOLLA
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
  #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
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
  #endif  // MPI_CHOLLA
      }
    }

    // Write out the xz datasets for each variable
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_d, "/d_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mx, "/mx_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_my, "/my_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mz, "/mz_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_E, "/E_xz");
  #ifdef MHD
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_xz");
  #endif  // MHD
  #ifdef DE
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_GE, "/GE_xz");
  #endif
  #ifdef SCALAR
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_scalar, "/scalar_xz");
  #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

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

    // Create the yz data space for the datasets
    dims[0]      = ny_dset;
    dims[1]      = nz_dset;
    dataspace_id = H5Screate_simple(2, dims, NULL);

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
        id     = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
        buf_id = k + j * H.nz_real;
  #ifdef MHD
        int id_xm1 = cuda_utilities::compute1DIndex(xslice - 1, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
        int id_ym1 = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost - 1, k + H.n_ghost, H.nx, H.ny);
        int id_zm1 = cuda_utilities::compute1DIndex(xslice, j + H.n_ghost, k + H.n_ghost - 1, H.nx, H.ny);
  #endif  // MHD
  #ifdef MPI_CHOLLA
        // When there are multiple processes, check whether this slice is in
        // your domain
        if (xslice >= nx_local_start && xslice < nx_local_start + nx_local) {
          id = cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
    #ifdef MHD
          int id_xm1 =
              cuda_utilities::compute1DIndex(xslice - nx_local_start - 1, j + H.n_ghost, k + H.n_ghost, H.nx, H.ny);
          int id_ym1 =
              cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost - 1, k + H.n_ghost, H.nx, H.ny);
          int id_zm1 =
              cuda_utilities::compute1DIndex(xslice - nx_local_start, j + H.n_ghost, k + H.n_ghost - 1, H.nx, H.ny);
    #endif  // MHD
  #endif    // MPI_CHOLLA
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
  #ifdef MPI_CHOLLA
        }
        // if the slice isn't in your domain, just write out zeros
        else {
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
  #endif  // MPI_CHOLLA
      }
    }

    // Write out the yz datasets for each variable
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_d, "/d_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mx, "/mx_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_my, "/my_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_mz, "/mz_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_E, "/E_yz");
  #ifdef MHD
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_x.data(), "/magnetic_x_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_y.data(), "/magnetic_y_yz");
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_magnetic_z.data(), "/magnetic_z_yz");
  #endif  // MHD
  #ifdef DE
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_GE, "/GE_yz");
  #endif
  #ifdef SCALAR
    status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer_scalar, "/scalar_yz");
  #endif

    // Free the dataspace id
    status = H5Sclose(dataspace_id);

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

  } else {
    printf("Slice write only works for 3D data.\n");
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