/*!
 * \file
 * Implements the ProjectionWriter type
 */

#include "ProjectionWriter.h"

#ifndef HDF5
  #include <hdf5.h>
#endif

#include <vector>

#include "../io/FnameTemplate.h"
#include "../io/io.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"
#include "../utils/hydro_utilities.h"

namespace io
{

namespace
{  // stuff inside an anonymous namespace is local to this file

/*! Write projected density and temperature data to a file, at the
 *  current simulation time. */
void Write_Projection_HDF5_(const Grid3D &G, hid_t file_id)
{
#ifndef HDF5
  CHOLLA_ERROR("this function must not be invoked when compiled without hdf5");
#else
  const Header &H = G.H;
  bool is_3D      = H.nx > 1 && H.ny > 1 && H.nz > 1;
  if (not is_3D) {
    chprintf("Projections only supported for 3D simulations.\n");
    return;
  }
  const Grid3D::Conserved &C = G.C;

  herr_t status;

  Real mu = 0.6;

  int nx_dset = H.nx_real;
  int ny_dset = H.ny_real;
  int nz_dset = H.nz_real;
  hsize_t dims[2];
  std::vector<Real> dataset_buffer_dxy(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_dxz(H.nx_real * H.nz_real);
  std::vector<Real> dataset_buffer_Txy(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_Txz(H.nx_real * H.nz_real);
  #ifdef DUST
  std::vector<Real> dataset_buffer_dust_xy(H.nx_real * H.ny_real);
  std::vector<Real> dataset_buffer_dust_xz(H.nx_real * H.nz_real);
  #endif

  // Create the data space for the datasets
  dims[0]               = nx_dset;
  dims[1]               = ny_dset;
  hid_t dataspace_xy_id = H5Screate_simple(2, dims, nullptr);
  dims[1]               = nz_dset;
  hid_t dataspace_xz_id = H5Screate_simple(2, dims, nullptr);

  // define a lambda function to compute temperature
  auto calc_T = [&](int xid, int yid, int zid) -> Real {
    int const id = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);

    Real const d = C.density[id];
    // calculate number density
    Real const n = d * DENSITY_UNIT / (mu * MP);

  // calculate temperature
  #ifdef DE
    Real const T = hydro_utilities::Calc_Temp_DE(C.GasEnergy[id], gama, n);
  #else  // DE is not defined
    Real const mx = C.momentum_x[id];
    Real const my = C.momentum_y[id];
    Real const mz = C.momentum_z[id];
    Real const E  = C.Energy[id];

    #ifdef MHD
    auto const magnetic_centered =
        mhd::utils::cellCenteredMagneticFields(C.host, id, xid, yid, zid, H.n_cells, H.nx, H.ny);
    Real const T = hydro_utilities::Calc_Temp_Conserved(E, d, mx, my, mz, gama, n, magnetic_centered.x(),
                                                        magnetic_centered.y(), magnetic_centered.z());
    #else   // MHD is not defined
    Real const T = hydro_utilities::Calc_Temp_Conserved(E, d, mx, my, mz, gama, n);
    #endif  // MHD
  #endif    // DE
    return T;
  };

  // in principle, we could refactor out some of this logic so that we could get rid of
  // the ifdef DUST statement. For now, that doesn't seem very urgent

  // Copy the xy density and temperature projections to the memory buffer
  for (int j = 0; j < H.ny_real; j++) {
    for (int i = 0; i < H.nx_real; i++) {
      Real dxy = 0;
      Real Txy = 0;
  #ifdef DUST
      Real dust_xy = 0;
  #endif
      // for each xy element, sum over the z column
      for (int k = 0; k < H.nz_real; k++) {
        int const xid = i + H.n_ghost;
        int const yid = j + H.n_ghost;
        int const zid = k + H.n_ghost;
        int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);

        // sum density
        Real const d = C.density[id];
        dxy += d * H.dz;
  #ifdef DUST
        dust_xy += C.dust_density[id] * H.dz;
  #endif
        Txy += calc_T(xid, yid, zid) * d * H.dz;
      }
      int const buf_id           = j + i * H.ny_real;
      dataset_buffer_dxy[buf_id] = dxy;
      dataset_buffer_Txy[buf_id] = Txy;
  #ifdef DUST
      dataset_buffer_dust_xy[buf_id] = dust_xy;
  #endif
    }
  }

  // Copy the xz density and temperature projections to the memory buffer
  for (int k = 0; k < H.nz_real; k++) {
    for (int i = 0; i < H.nx_real; i++) {
      Real dxz = 0;
      Real Txz = 0;
  #ifdef DUST
      Real dust_xz = 0;
  #endif
      // for each xz element, sum over the y column
      for (int j = 0; j < H.ny_real; j++) {
        int const xid = i + H.n_ghost;
        int const yid = j + H.n_ghost;
        int const zid = k + H.n_ghost;
        int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);
        // sum density
        Real const d = C.density[id];
        dxz += d * H.dy;
  #ifdef DUST
        dust_xz += C.dust_density[id] * H.dy;
  #endif
        Txz += calc_T(xid, yid, zid) * d * H.dy;
      }
      int const buf_id           = k + i * H.nz_real;
      dataset_buffer_dxz[buf_id] = dxz;
      dataset_buffer_Txz[buf_id] = Txz;
  #ifdef DUST
      dataset_buffer_dust_xz[buf_id] = dust_xz;
  #endif
    }
  }

  // Write the projected density and temperature arrays to file
  status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_dxy.data(), "/d_xy");
  status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_dxz.data(), "/d_xz");
  status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_Txy.data(), "/T_xy");
  status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_Txz.data(), "/T_xz");
  #ifdef DUST
  status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_dust_xy.data(), "/d_dust_xy");
  status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_dust_xz.data(), "/d_dust_xz");
  #endif

  // Free the dataspace ids
  status = H5Sclose(dataspace_xz_id);
  status = H5Sclose(dataspace_xy_id);
#endif  // HDF5
}

}  // anonymous namespace

void ProjectionWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const
{
#ifdef HDF5
  // create the filename
  std::string filename = fname_template.format_fname(nfile, "_proj");

  // Create a new file
  hid_t file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write the density and temperature projections to the output file
  Write_Projection_HDF5_(G, file_id);

  // Close the file
  if (H5Fclose(file_id) < 0) {
    CHOLLA_ERROR("File write failed. ProcID: %d\n", procID);
  }

#else
  chprintf("Output_Projected_Data only defined for hdf5 writes.\n");
#endif  // HDF5
}

}  // namespace io