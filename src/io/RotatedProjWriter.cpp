/*!
 * \file
 * Implements the RotatedProjWriter type
 */

#include "RotatedProjWriter.h"

#include <cmath>  // M_PI (note: not guaranteed by the C++ standard)
#ifdef HDF5
  #include <hdf5.h>
#endif  // HDF5

#include "../global/global.h"  // Parameters
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../utils/cuda_utilities.h"
#include "../utils/error_handling.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"

namespace
{  // contents of an anonymous namespace are local to current translation unit

/*! function used to rotate points about an axis in 3D for the rotated projection
 *  output routine */
void Rotate_Point(Real x, Real y, Real z, Real delta, Real phi, Real theta, Real *xp, Real *yp, Real *zp)
{
  Real cd, sd, cp, sp, ct, st;  // sines and cosines
  Real a00, a01, a02;           // rotation matrix elements
  Real a10, a11, a12;
  Real a20, a21, a22;

  // compute trig functions of rotation angles
  cd = cos(delta);
  sd = sin(delta);
  cp = cos(phi);
  sp = sin(phi);
  ct = cos(theta);
  st = sin(theta);

  // compute the rotation matrix elements
  /*a00 =       cosp*cosd - sinp*cost*sind;
  a01 = -1.0*(cosp*sind + sinp*cost*cosd);
  a02 =       sinp*sint;

  a10 =       sinp*cosd + cosp*cost*sind;
  a11 =      (cosp*cost*cosd - sint*sind);
  a12 = -1.0* cosp*sint;

  a20 =       sint*sind;
  a21 =       sint*cosd;
  a22 =       cost;*/
  a00 = (cp * cd - sp * ct * sd);
  a01 = -1.0 * (cp * sd + sp * ct * cd);
  a02 = sp * st;
  a10 = (sp * cd + cp * ct * sd);
  a11 = (cp * ct * cd - st * sd);
  a12 = cp * st;
  a20 = st * sd;
  a21 = st * cd;
  a22 = ct;

  *xp = a00 * x + a01 * y + a02 * z;
  *yp = a10 * x + a11 * y + a12 * z;
  *zp = a20 * x + a21 * y + a22 * z;
}

/*! \brief Write rotated projected data to a file, at the current simulation */
void Write_Rotated_Projection_HDF5_(const Grid3D &G, hid_t file_id, const io::Rotation &R)
{
#ifndef HDF5
  CHOLLA_ERROR("this function must not get called when compiled without hdf5")
#else
  const Header &H = G.H;
  bool is_3D      = H.nx > 1 && H.ny > 1 && H.nz > 1;
  if (not is_3D) {
    chprintf("Slice write only works for 3D data.\n");
    return;
  }

  const FieldInfo &field_info = G.field_info;
  const Grid3D::Conserved &C  = G.C;
  hid_t dataset_id, dataspace_xzr_id;
  Real *dataset_buffer_dxzr;
  Real *dataset_buffer_Txzr;
  Real *dataset_buffer_vxxzr;
  Real *dataset_buffer_vyxzr;
  Real *dataset_buffer_vzxzr;

  herr_t status;
  Real dxy, dxz, Txy, Txz;
  Real d, vx, vy, vz;

  Real x, y, z;      // cell positions
  Real xp, yp, zp;   // rotated positions
  Real alpha, beta;  // projected positions
  int ix, iz;        // projected index positions

  Real mu = 0.6;

  srand(137);      // initialize a random number
  Real eps = 0.1;  // randomize cell centers slightly to combat aliasing

  // 3D
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    Real Lx     = R.Lx;  // projected box size in x dir
    Real Lz     = R.Lz;  // projected box size in z dir
    int nx_dset = R.nx;
    int nz_dset = R.nz;

    if (R.nx * R.nz == 0) {
      chprintf(
          "WARNING: compiled with -DROTATED_PROJECTION but input parameters "
          "nxr or nzr = 0\n");
      return;
    }

    // set the projected dataset size for this process to capture
    // this piece of the simulation volume
    // min and max values were set in the header write
    int nx_min, nx_max, nz_min, nz_max;
    nx_min  = R.nx_min;
    nx_max  = R.nx_max;
    nz_min  = R.nz_min;
    nz_max  = R.nz_max;
    nx_dset = nx_max - nx_min;
    nz_dset = nz_max - nz_min;

    hsize_t dims[2];

    // allocate the buffers for the projected dataset
    // and initialize to zero
    dataset_buffer_dxzr  = (Real *)calloc(nx_dset * nz_dset, sizeof(Real));
    dataset_buffer_Txzr  = (Real *)calloc(nx_dset * nz_dset, sizeof(Real));
    dataset_buffer_vxxzr = (Real *)calloc(nx_dset * nz_dset, sizeof(Real));
    dataset_buffer_vyxzr = (Real *)calloc(nx_dset * nz_dset, sizeof(Real));
    dataset_buffer_vzxzr = (Real *)calloc(nx_dset * nz_dset, sizeof(Real));

    // Create the data space for the datasets
    dims[0]          = nx_dset;
    dims[1]          = nz_dset;
    dataspace_xzr_id = H5Screate_simple(2, dims, NULL);

    // Copy the xz rotated projection to the memory buffer
    for (int k = 0; k < H.nz_real; k++) {
      for (int i = 0; i < H.nx_real; i++) {
        for (int j = 0; j < H.ny_real; j++) {
          // get cell index
          int const xid = i + H.n_ghost;
          int const yid = j + H.n_ghost;
          int const zid = k + H.n_ghost;
          int const id  = cuda_utilities::compute1DIndex(xid, yid, zid, H.nx, H.ny);

          // get cell positions
          G.Get_Position(i + H.n_ghost, j + H.n_ghost, k + H.n_ghost, &x, &y, &z);

          // add very slight noise to locations
          x += eps * H.dx * (drand48() - 0.5);
          y += eps * H.dy * (drand48() - 0.5);
          z += eps * H.dz * (drand48() - 0.5);

          // rotate cell positions
          Rotate_Point(x, y, z, R.delta, R.phi, R.theta, &xp, &yp, &zp);

          // find projected locations
          // assumes box centered at [0,0,0]
          alpha = (R.nx * (xp + 0.5 * R.Lx) / R.Lx);
          beta  = (R.nz * (zp + 0.5 * R.Lz) / R.Lz);
          ix    = (int)round(alpha);
          iz    = (int)round(beta);
  #ifdef MPI_CHOLLA
          ix = ix - nx_min;
          iz = iz - nz_min;
  #endif

          if ((ix >= 0) && (ix < nx_dset) && (iz >= 0) && (iz < nz_dset)) {
            int const buf_id = iz + ix * nz_dset;
            d                = C.density[id];
            // project density
            dataset_buffer_dxzr[buf_id] += d * H.dy;
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

            Txz = T * d * H.dy;
            dataset_buffer_Txzr[buf_id] += Txz;

            // compute velocities
            dataset_buffer_vxxzr[buf_id] += C.momentum_x[id] * H.dy;
            dataset_buffer_vyxzr[buf_id] += C.momentum_y[id] * H.dy;
            dataset_buffer_vzxzr[buf_id] += C.momentum_z[id] * H.dy;
          }
        }
      }
    }

    // Write projected d,T,vx,vy,vz
    status = Write_HDF5_Dataset(file_id, dataspace_xzr_id, dataset_buffer_dxzr, "/d_xzr");
    status = Write_HDF5_Dataset(file_id, dataspace_xzr_id, dataset_buffer_Txzr, "/T_xzr");
    status = Write_HDF5_Dataset(file_id, dataspace_xzr_id, dataset_buffer_vxxzr, "/vx_xzr");
    status = Write_HDF5_Dataset(file_id, dataspace_xzr_id, dataset_buffer_vyxzr, "/vy_xzr");
    status = Write_HDF5_Dataset(file_id, dataspace_xzr_id, dataset_buffer_vzxzr, "/vz_xzr");

    // Free the dataspace id
    status = H5Sclose(dataspace_xzr_id);

    // free the data
    free(dataset_buffer_dxzr);
    free(dataset_buffer_Txzr);
    free(dataset_buffer_vxxzr);
    free(dataset_buffer_vyxzr);
    free(dataset_buffer_vzxzr);

  } else {
    chprintf("Rotated projection write only implemented for 3D data.\n");
  }

#endif  // HDF5
}

/*! Write the relevant header info to the HDF5 file for rotated projection.
 *
 *  \todo Make G a const argument to ensure we don't mutate it
 */
void Write_Header_Rotated_(Grid3D &G, hid_t file_id, io::Rotation &R)
{
#ifndef HDF5
  CHOLLA_ERROR("this function must not get called when compiled without hdf5")
#else
  const Header &H = G.H;
  G.Write_Header_HDF5(file_id, true);
  Real delta, theta, phi;

  #ifdef MPI_CHOLLA
  // determine the size of the projection to output for this subvolume
  Real x, y, z, xp, yp, zp;
  Real alpha, beta;
  int ix, iz;
  R.nx_min = R.nx;
  R.nx_max = 0;
  R.nz_min = R.nz;
  R.nz_max = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        // find the corners of this domain in the rotated position
        G.Get_Position(H.n_ghost + i * (H.nx - 2 * H.n_ghost), H.n_ghost + j * (H.ny - 2 * H.n_ghost),
                       H.n_ghost + k * (H.nz - 2 * H.n_ghost), &x, &y, &z);
        // rotate cell position
        Rotate_Point(x, y, z, R.delta, R.phi, R.theta, &xp, &yp, &zp);
        // find projected location
        // assumes box centered at [0,0,0]
        alpha    = (R.nx * (xp + 0.5 * R.Lx) / R.Lx);
        beta     = (R.nz * (zp + 0.5 * R.Lz) / R.Lz);
        ix       = (int)round(alpha);
        iz       = (int)round(beta);
        R.nx_min = std::min(ix, R.nx_min);
        R.nx_max = std::max(ix, R.nx_max);
        R.nz_min = std::min(iz, R.nz_min);
        R.nz_max = std::max(iz, R.nz_max);
      }
    }
  }
  // if the corners aren't within the chosen projection area
  // take the input projection edge as the edge of this piece of the projection
  R.nx_min = std::max(R.nx_min, 0);
  R.nx_max = std::min(R.nx_max, R.nx);
  R.nz_min = std::max(R.nz_min, 0);
  R.nz_max = std::min(R.nz_max, R.nz);
  #endif

  H5AttrRecorder attr_recorder(file_id);

  // Rotation data
  attr_recorder.record("nxr", R.nx);
  attr_recorder.record("nzr", R.nz);
  attr_recorder.record("nx_min", R.nx_min);
  attr_recorder.record("nz_min", R.nz_min);
  attr_recorder.record("nx_max", R.nx_max);
  attr_recorder.record("nz_max", R.nz_max);
  delta = 180. * R.delta / M_PI;
  attr_recorder.record("delta", delta);
  theta = 180. * R.theta / M_PI;
  attr_recorder.record("theta", theta);
  phi = 180. * R.phi / M_PI;
  attr_recorder.record("phi", phi);
  attr_recorder.record("Lx", R.Lx);
  attr_recorder.record("Lz", R.Lz);

  chprintf(
      "Outputting rotation data with delta = %e, theta = %e, phi = %e, Lx = "
      "%f, Lz = %f\n",
      R.delta, R.theta, R.phi, R.Lx, R.Lz);
#endif  // HDF5
}

}  // anonymous namespace

io::Rotation::Rotation(ParameterMap &pmap)
{
  // thoughts for the future:
  // do we want to enforce valid domains for arguments? We might require that
  // - nxr & nzr are positive
  // - delta, phi, & theta satisfy 0 <= x < 180 (maybe we want to be more flexible)
  // - Lx & Lz are positive (or at least non-zero)
  // - I think flag_delta should only be 0, 1, or 2
  //
  // should we make it an error to specify the:
  // - ddelta_dt parameter when flag_delta != 1
  // - n_delta parameter when flag_delta != 2

  // x-dir pixels in projection
  this->nx = pmap.value<int>("nxr");
  // z-dir pixels in projection
  this->nz = pmap.value<int>("nzr");
  // minimum x location to project
  this->nx_min = 0;
  // minimum z location to project
  this->nz_min = 0;
  // maximum x location to project
  this->nx_max = this->nx;
  // maximum z location to project
  this->nz_max = this->nz;
  // rotation angle about z direction
  this->delta = Real(M_PI * (pmap.value_or("delta", 0.0) / 180.));  // convert to radians
  // rotation angle about x direction
  this->theta = Real(M_PI * (pmap.value_or("theta", 0.0) / 180.));  // convert to radians
  // rotation angle about y direction
  this->phi = Real(M_PI * (pmap.value_or("phi", 0.0) / 180.));  // convert to radians
  // x-dir physical size of projection
  this->Lx = Real(pmap.value<double>("Lx"));
  // z-dir physical size of projection
  this->Lz = Real(pmap.value<double>("Lz"));
  // initialize a counter for rotated outputs
  this->i_delta = 0;
  // number of rotated outputs in a complete revolution
  this->n_delta = pmap.value_or("n_delta", 0);
  CHOLLA_ASSERT(this->n_delta >= 0, "the \"n_delta\" parameter must not be negative");
  // rate of rotation between outputs, for an actual simulation
  this->ddelta_dt = Real(pmap.value_or("ddelta_dt", 0.0));
  // are we not rotating about z(0)?
  // are we outputting multiple rotations(1)? or rotating during a
  // simulation(2)?
  this->flag_delta = pmap.value_or("flag_delta", 0);

  // after we make it possible to enable the RotatedProjectionWriter without a
  // compile-time ifdef, the following should probably get converted to an error
  // that aborts the program
  if (flag_delta == 1 && n_delta == 0) {
    chprintf(
        "WARNING: when flag_delta = 1 and n_delta = 0, no rotated projections "
        "are made\n");
  }
}

void io::RotatedProjWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template)
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  std::string_view standard_suffix = "_rot_proj";

  // it may be a little more explicit to use a switch statement instead of if/elif/else
  switch (this->rot_info_.flag_delta) {
    case 1: {
      // if flag_delta==1, then we are just outputting a
      // bunch of rotations of the same snapshot

      for (int i_delta = 0; i_delta < this->rot_info_.n_delta; i_delta++) {
        // determine the filename
        std::string post_extension_suffix = std::to_string(i_delta);
        std::string filename =
            fname_template.format_fname(nfile, standard_suffix, std::optional<std::string_view>(post_extension_suffix));

        // determine delta about z by output index
        this->rot_info_.delta = 2.0 * M_PI * ((double)i_delta) / ((double)this->rot_info_.n_delta);

        // Create a new file
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        // Write header (file attributes)
        Write_Header_Rotated_(G, file_id, this->rot_info_);

        // Write the density and temperature projections to the output file
        Write_Rotated_Projection_HDF5_(G, file_id, this->rot_info_);

        // Close the file
        status = H5Fclose(file_id);

        CHOLLA_ASSERT(status >= 0, "Rotated Projection: File write failed. ProcID: %d\n", procID);

        // iterate this->rot_info_.i_delta
        this->rot_info_.i_delta++;
      }
      break;
    }
    case 2: {  // outputing at a rotating delta
      // determine the filename
      std::string filename = fname_template.format_fname(nfile, standard_suffix);

      // rotation rate given in the parameter file
      this->rot_info_.delta = fmod(nfile * this->rot_info_.ddelta_dt * 2.0 * M_PI, (2.0 * M_PI));

      // Create a new file
      file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write header (file attributes)
      Write_Header_Rotated_(G, file_id, this->rot_info_);

      // Write the density and temperature projections to the output file
      Write_Rotated_Projection_HDF5_(G, file_id, this->rot_info_);

      // Close the file
      status = H5Fclose(file_id);
      break;
    }
    case 0: {  // just output at the delta given in the parameter file
      // determine the filename
      std::string filename = fname_template.format_fname(nfile, standard_suffix);

      // Create a new file
      file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write header (file attributes)
      Write_Header_Rotated_(G, file_id, this->rot_info_);

      // Write the density and temperature projections to the output file
      Write_Rotated_Projection_HDF5_(G, file_id, this->rot_info_);

      // Close the file
      status = H5Fclose(file_id);
      break;
    }
    default:
      CHOLLA_ERROR("Invalid flag_delta: %d", this->rot_info_.flag_delta);
  }

  CHOLLA_ASSERT(status >= 0, "Rotated Projection: File write failed. ProcID: %d\n", procID);

#else
  printf("Output_Rotated_Projected_Data only defined for HDF5 writes.\n");
#endif
}