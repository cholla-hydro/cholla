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

io::Rotation::Rotation(ParameterMap &pmap)
{
  // thoughts for the future:
  // do we want to enforce valid domains for arguments? We might require that
  // - nxr & nzr are positive
  // - delta, phi, & theta satisfy 0 <= x < 180 (maybe we want to be more flexible)
  // - Lx & Lz are positive (or at least non-zero)
  // - I think flag_delta should only be 0, 1, or 2

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
  // rate of rotation between outputs, for an actual simulation
  this->ddelta_dt = Real(pmap.value_or("ddelta_dt", 0.0));
  // are we not rotating about z(0)?
  // are we outputting multiple rotations(1)? or rotating during a
  // simulation(2)?
  this->flag_delta = pmap.value_or("flag_delta", 0);
}

void io::RotatedProjWriter::operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template)
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename = fname_template.format_fname(nfile, "_rot_proj");

  if (this->rot_info_.flag_delta == 1) {
    // if flag_delta==1, then we are just outputting a
    // bunch of rotations of the same snapshot
    int i_delta;
    char fname[200];

    for (i_delta = 0; i_delta < this->rot_info_.n_delta; i_delta++) {
      filename += "." + std::to_string(this->rot_info_.i_delta);
      chprintf("Outputting rotated projection %s.\n", fname);

      // determine delta about z by output index
      this->rot_info_.delta = 2.0 * M_PI * ((double)i_delta) / ((double)this->rot_info_.n_delta);

      // Create a new file
      file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write header (file attributes)
      G.Write_Header_Rotated_HDF5(file_id, this->rot_info_);

      // Write the density and temperature projections to the output file
      G.Write_Rotated_Projection_HDF5(file_id, this->rot_info_);

      // Close the file
      status = H5Fclose(file_id);
  #ifdef MPI_CHOLLA
      if (status < 0) {
        printf("Output_Rotated_Projected_Data: File write failed. ProcID: %d\n", procID);
        chexit(-1);
      }
  #else
      if (status < 0) {
        printf("Output_Rotated_Projected_Data: File write failed.\n");
        exit(-1);
      }
  #endif

      // iterate this->rot_info_.i_delta
      this->rot_info_.i_delta++;
    }

  } else if (this->rot_info_.flag_delta == 2) {
    // case 2 -- outputting at a rotating delta
    // rotation rate given in the parameter file
    this->rot_info_.delta = fmod(nfile * this->rot_info_.ddelta_dt * 2.0 * M_PI, (2.0 * M_PI));

    // Create a new file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id, this->rot_info_);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id, this->rot_info_);

    // Close the file
    status = H5Fclose(file_id);
  } else {
    // case 0 -- just output at the delta given in the parameter file

    // Create a new file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id, this->rot_info_);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id, this->rot_info_);

    // Close the file
    status = H5Fclose(file_id);
  }

  #ifdef MPI_CHOLLA
  if (status < 0) {
    printf("Output_Rotated_Projected_Data: File write failed. ProcID: %d\n", procID);
    chexit(-1);
  }
  #else
  if (status < 0) {
    printf("Output_Rotated_Projected_Data: File write failed.\n");
    exit(-1);
  }
  #endif

#else
  printf("Output_Rotated_Projected_Data only defined for HDF5 writes.\n");
#endif
}