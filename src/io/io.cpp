#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#ifdef HDF5
  #include <hdf5.h>
#endif  // HDF5
#include "../grid/grid3D.h"
#include "../io/io.h"
#include "../utils/cuda_utilities.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"
#include "../utils/timing_functions.h"  // provides ScopedTimer
#ifdef MPI_CHOLLA
  #include "../mpi/mpi_routines.h"
#endif  // MPI_CHOLLA
#include "../utils/DeviceVector.h"
#include "../utils/error_handling.h"

#ifdef COSMOLOGY
  #include "../cosmology/cosmology.h"
#endif  // COSMOLOGY

// #define OUTPUT_ENERGY
// #define OUTPUT_MOMENTUM

/* function used to rotate points about an axis in 3D for the rotated projection
 * output routine */
void Rotate_Point(Real x, Real y, Real z, Real delta, Real phi, Real theta, Real *xp, Real *yp, Real *zp);

/* local function that designates whether we are using a root-process. It gives
 * gives a sensible result regardless of whether we are using MPI */
static inline bool Is_Root_Proc()
{
#ifdef MPI_CHOLLA
  return procID == root;
#else
  return true;
#endif
}

void Create_Log_File(struct Parameters P)
{
  if (not Is_Root_Proc()) {
    return;
  }

  std::string file_name(LOG_FILE_NAME);
  chprintf("\nCreating Log File: %s \n\n", file_name.c_str());

  bool file_exists = false;
  if (FILE *file = fopen(file_name.c_str(), "r")) {
    file_exists = true;
    chprintf("  File exists, appending values: %s \n\n", file_name.c_str());
    fclose(file);
  }

  // current date/time based on current system
  time_t now = time(0);
  // convert now to string form
  char *dt = ctime(&now);

  std::ofstream out_file;
  out_file.open(file_name.c_str(), std::ios::app);
  out_file << "\n";
  out_file << "Run date: " << dt;
  out_file.close();
}

void Write_Message_To_Log_File(const char *message)
{
  if (not Is_Root_Proc()) {
    return;
  }

  std::string file_name(LOG_FILE_NAME);
  std::ofstream out_file;
  out_file.open(file_name.c_str(), std::ios::app);
  out_file << message << std::endl;
  out_file.close();
}

/* Write Cholla Output Data */
void Write_Data(Grid3D &G, struct Parameters P, int nfile)
{
  cudaMemcpy(G.C.density, G.C.device, G.H.n_fields * G.H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);

  chprintf("\nSaving Snapshot: %d \n", nfile);

#ifdef HDF5
  // Initialize HDF5 interface
  H5open();
#endif

#ifdef N_OUTPUT_COMPLETE
  // If nfile is multiple of N_OUTPUT_COMPLETE then output all data
  if (nfile % N_OUTPUT_COMPLETE == 0) {
    G.H.Output_Complete_Data = true;
    chprintf(" Writing all data ( Restart File ).\n");
  } else {
    G.H.Output_Complete_Data = false;
  }

#else
  // If NOT N_OUTPUT_COMPLETE: always output complete data
  G.H.Output_Complete_Data = true;
#endif

#ifdef COSMOLOGY
  G.Change_Cosmological_Frame_Sytem(false);
#endif

#ifndef ONLY_PARTICLES
  /*call the data output routine for Hydro data*/
  if (nfile % P.n_hydro == 0) {
    Output_Data(G, P, nfile);
  }
#endif

// This function does other checks to make sure it is valid (3D only)
#ifdef HDF5
  if (P.n_out_float32 && nfile % P.n_out_float32 == 0) {
    Output_Float32(G, P, nfile);
  }
#endif

#ifdef PROJECTION
  if (nfile % P.n_projection == 0) {
    Output_Projected_Data(G, P, nfile);
  }
#endif /*PROJECTION*/

#ifdef ROTATED_PROJECTION
  if (nfile % P.n_rotated_projection == 0) {
    Output_Rotated_Projected_Data(G, P, nfile);
  }
#endif /*ROTATED_PROJECTION*/

#ifdef SLICES
  if (nfile % P.n_slice == 0) {
    Output_Slices(G, P, nfile);
  }
#endif /*SLICES*/

#ifdef PARTICLES
  if (nfile % P.n_particle == 0) {
    G.WriteData_Particles(P, nfile);
  }
#endif

#ifdef COSMOLOGY
  if (G.H.OUTPUT_SCALE_FACOR || G.H.Output_Initial) {
    G.Cosmo.Set_Next_Scale_Output();
    if (!G.Cosmo.exit_now) {
      chprintf(" Saved Snapshot: %d     z:%f   next_output: %f\n", nfile, G.Cosmo.current_z,
               1 / G.Cosmo.next_output - 1);
      G.H.Output_Initial = false;
    } else {
      chprintf(" Saved Snapshot: %d     z:%f   Exiting now\n", nfile, G.Cosmo.current_z);
    }

  } else {
    chprintf(" Saved Snapshot: %d     z:%f\n", nfile, G.Cosmo.current_z);
  }
  G.Change_Cosmological_Frame_Sytem(true);
  chprintf("\n");
  G.H.Output_Now = false;
#endif

#ifdef HDF5
  // Cleanup HDF5
  H5close();
#endif

#if defined(GRAVITY) && defined(HDF5)
  G.Grav.Write_Restart_HDF5(&P, nfile);
#endif

#ifdef MPI_CHOLLA
  MPI_Barrier(world);
#endif
}

/* Output the grid data to file. */
void Output_Data(Grid3D &G, struct Parameters P, int nfile)
{
  // create the filename
  std::string filename(P.outdir);
  filename += std::to_string(nfile);

#if defined BINARY
  filename += ".bin";
#elif defined HDF5
  filename += ".h5";
#else
  filename += ".txt";
  if (G.H.nx * G.H.ny * G.H.nz > 1000) printf("Ascii outputs only recommended for small problems!\n");
#endif
#ifdef MPI_CHOLLA
  filename += "." + std::to_string(procID);
#endif

// open the file for binary writes
#if defined BINARY
  FILE *out;
  out = fopen(filename.data(), "w");
  if (out == NULL) {
    printf("Error opening output file.\n");
    exit(-1);
  }

  // write the header to the output file
  G.Write_Header_Binary(out);

  // write the conserved variables to the output file
  G.Write_Grid_Binary(out);

  // close the output file
  fclose(out);

// create the file for hdf5 writes
#elif defined HDF5
  hid_t file_id; /* file identifier */
  herr_t status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file
  G.Write_Grid_HDF5(file_id);

  // close the file
  status = H5Fclose(file_id);

  if (status < 0) {
    printf("File write failed.\n");
    exit(-1);
  }

#else
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

void Output_Float32(Grid3D &G, struct Parameters P, int nfile)
{
#ifdef HDF5
  Header H = G.H;
  // Do nothing in 1-D and 2-D case
  if (H.ny_real == 1) {
    return;
  }
  if (H.nz_real == 1) {
    return;
  }
  // Do nothing if nfile is not multiple of n_out_float32
  if (nfile % P.n_out_float32 != 0) {
    return;
  }

  // create the filename
  std::string filename(P.outdir);
  filename += std::to_string(nfile);
  filename += ".float32.h5";
  #ifdef MPI_CHOLLA
  filename += "." + std::to_string(procID);
  #endif  // MPI_CHOLLA

  // create hdf5 file
  hid_t file_id; /* file identifier */
  herr_t status;

  // Create a new file using default properties.
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write the header (file attributes)
  G.Write_Header_HDF5(file_id);

  // write the conserved variables to the output file

  // 3-D Case
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    int nx_dset = H.nx_real;
    int ny_dset = H.ny_real;
    int nz_dset = H.nz_real;
    size_t buffer_size;
    // Need a larger device buffer for MHD. In the future, if other fields need
    // a larger device buffer, choose the maximum of the sizes. If the buffer is
    // too large, it does not cause bugs (Oct 6 2022)
  #ifdef MHD
    buffer_size = (nx_dset + 1) * (ny_dset + 1) * (nz_dset + 1);
  #else
    buffer_size = nx_dset * ny_dset * nz_dset;
  #endif  // MHD

    // Using static DeviceVector here automatically allocates the buffer the
    // first time it is needed It persists until program exit, and then calls
    // Free upon destruction
    cuda_utilities::DeviceVector<float> static device_dataset_vector{buffer_size};
    auto *dataset_buffer = (float *)malloc(buffer_size * sizeof(float));

    if (P.out_float32_density > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_density, "/density");
    }
    if (P.out_float32_momentum_x > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_momentum_x, "/momentum_x");
    }
    if (P.out_float32_momentum_y > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_momentum_y, "/momentum_y");
    }
    if (P.out_float32_momentum_z > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_momentum_z, "/momentum_z");
    }
    if (P.out_float32_Energy > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_Energy, "/Energy");
    }
  #ifdef DE
    if (P.out_float32_GasEnergy > 0) {
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset, ny_dset, nz_dset, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_GasEnergy, "/GasEnergy");
    }
  #endif  // DE
  #ifdef MHD

    // TODO (by Alwin, for anyone) : Repair output format if needed and remove these chprintfs when appropriate
    if (P.out_float32_magnetic_x > 0) {
      chprintf("WARNING: MHD float-32 output has a different output format than float-64\n");
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset + 1, ny_dset + 1, nz_dset + 1, H.n_ghost - 1, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_magnetic_x, "/magnetic_x");
    }
    if (P.out_float32_magnetic_y > 0) {
      chprintf("WARNING: MHD float-32 output has a different output format than float-64\n");
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset + 1, ny_dset + 1, nz_dset + 1, H.n_ghost - 1, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_magnetic_y, "/magnetic_y");
    }
    if (P.out_float32_magnetic_z > 0) {
      chprintf("WARNING: MHD float-32 output has a different output format than float-64\n");
      Write_HDF5_Field_3D(H.nx, H.ny, nx_dset + 1, ny_dset + 1, nz_dset + 1, H.n_ghost - 1, file_id, dataset_buffer,
                          device_dataset_vector.data(), G.C.d_magnetic_z, "/magnetic_z");
    }

  #endif  // MHD

    free(dataset_buffer);

    if (status < 0) {
      printf("File write failed.\n");
      exit(-1);
    }
  }  // 3-D case

  // close the file
  status = H5Fclose(file_id);
#endif  // HDF5
}

/* Output a projection of the grid data to file. */
void Output_Projected_Data(Grid3D &G, struct Parameters P, int nfile)
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename(P.outdir);
  filename += std::to_string(nfile);
  filename += "_proj.h5";

  #ifdef MPI_CHOLLA
  filename += "." + std::to_string(procID);
  #endif /*MPI_CHOLLA*/

  // Create a new file
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write the density and temperature projections to the output file
  G.Write_Projection_HDF5(file_id);

  // Close the file
  status = H5Fclose(file_id);

  #ifdef MPI_CHOLLA
  if (status < 0) {
    printf("Output_Projected_Data: File write failed. ProcID: %d\n", procID);
    chexit(-1);
  }
  #else
  if (status < 0) {
    printf("Output_Projected_Data: File write failed.\n");
    exit(-1);
  }
  #endif

#else
  printf("Output_Projected_Data only defined for hdf5 writes.\n");
#endif  // HDF5
}

/* Output a rotated projection of the grid data to file. */
void Output_Rotated_Projected_Data(Grid3D &G, struct Parameters P, int nfile)
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename(P.outdir);
  filename += std::to_string(nfile);
  filename += "_rot_proj.h5";

  #ifdef MPI_CHOLLA
  filename += "." + std::to_string(procID);
  #endif /*MPI_CHOLLA*/

  if (G.R.flag_delta == 1) {
    // if flag_delta==1, then we are just outputting a
    // bunch of rotations of the same snapshot
    int i_delta;
    char fname[200];

    for (i_delta = 0; i_delta < G.R.n_delta; i_delta++) {
      filename += "." + std::to_string(G.R.i_delta);
      chprintf("Outputting rotated projection %s.\n", fname);

      // determine delta about z by output index
      G.R.delta = 2.0 * M_PI * ((double)i_delta) / ((double)G.R.n_delta);

      // Create a new file
      file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

      // Write header (file attributes)
      G.Write_Header_Rotated_HDF5(file_id);

      // Write the density and temperature projections to the output file
      G.Write_Rotated_Projection_HDF5(file_id);

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

      // iterate G.R.i_delta
      G.R.i_delta++;
    }

  } else if (G.R.flag_delta == 2) {
    // case 2 -- outputting at a rotating delta
    // rotation rate given in the parameter file
    G.R.delta = fmod(nfile * G.R.ddelta_dt * 2.0 * M_PI, (2.0 * M_PI));

    // Create a new file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id);

    // Close the file
    status = H5Fclose(file_id);
  } else {
    // case 0 -- just output at the delta given in the parameter file

    // Create a new file
    file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Write header (file attributes)
    G.Write_Header_Rotated_HDF5(file_id);

    // Write the density and temperature projections to the output file
    G.Write_Rotated_Projection_HDF5(file_id);

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

/* Output xy, xz, and yz slices of the grid data. */
void Output_Slices(Grid3D &G, struct Parameters P, int nfile)
{
#ifdef HDF5
  hid_t file_id;
  herr_t status;

  // create the filename
  std::string filename(P.outdir);
  filename += std::to_string(nfile);
  filename += "_slice.h5";

  #ifdef MPI_CHOLLA
  filename += "." + std::to_string(procID);
  #endif /*MPI_CHOLLA*/

  // Create a new file
  file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write header (file attributes)
  G.Write_Header_HDF5(file_id);

  // Write slices of all variables to the output file
  G.Write_Slices_HDF5(file_id);

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

/*! \fn void Write_Header_Text(FILE *fp)
 *  \brief Write some relevant header info to a text output file. */
void Grid3D::Write_Header_Text(FILE *fp)
{
  // Write the header info to the output file
  fprintf(fp, "Header Information\n");
  fprintf(fp, "Git Commit Hash = %s\n", GIT_HASH);
  fprintf(fp, "Macro Flags     = %s\n", MACRO_FLAGS);
  fprintf(fp, "n_step: %d  sim t: %f  sim dt: %f\n", H.n_step, H.t, H.dt);
  fprintf(fp, "mass unit: %e  length unit: %e  time unit: %e\n", MASS_UNIT, LENGTH_UNIT, TIME_UNIT);
  fprintf(fp, "nx: %d  ny: %d  nz: %d\n", H.nx, H.ny, H.nz);
  fprintf(fp, "xmin: %f  ymin: %f  zmin: %f\n", H.xbound, H.ybound, H.zbound);
  fprintf(fp, "t: %f\n", H.t);
}

/*! \fn void Write_Header_Binary(FILE *fp)
 *  \brief Write the relevant header info to a binary output file. */
void Grid3D::Write_Header_Binary(FILE *fp)
{
  // Write the header info to the output file
  // fwrite(&H, sizeof(H), 1, fp);
  fwrite(&H.n_cells, sizeof(int), 1, fp);
  fwrite(&H.n_ghost, sizeof(int), 1, fp);
  fwrite(&H.nx, sizeof(int), 1, fp);
  fwrite(&H.ny, sizeof(int), 1, fp);
  fwrite(&H.nz, sizeof(int), 1, fp);
  fwrite(&H.nx_real, sizeof(int), 1, fp);
  fwrite(&H.ny_real, sizeof(int), 1, fp);
  fwrite(&H.nz_real, sizeof(int), 1, fp);
  fwrite(&H.xbound, sizeof(Real), 1, fp);
  fwrite(&H.ybound, sizeof(Real), 1, fp);
  fwrite(&H.zbound, sizeof(Real), 1, fp);
  fwrite(&H.xblocal, sizeof(Real), 1, fp);
  fwrite(&H.yblocal, sizeof(Real), 1, fp);
  fwrite(&H.zblocal, sizeof(Real), 1, fp);
  fwrite(&H.xdglobal, sizeof(Real), 1, fp);
  fwrite(&H.ydglobal, sizeof(Real), 1, fp);
  fwrite(&H.zdglobal, sizeof(Real), 1, fp);
  fwrite(&H.dx, sizeof(Real), 1, fp);
  fwrite(&H.dy, sizeof(Real), 1, fp);
  fwrite(&H.dz, sizeof(Real), 1, fp);
  fwrite(&H.t, sizeof(Real), 1, fp);
  fwrite(&H.dt, sizeof(Real), 1, fp);
  fwrite(&H.t_wall, sizeof(Real), 1, fp);
  fwrite(&H.n_step, sizeof(int), 1, fp);
}

#ifdef HDF5
/*! \fn void Write_Header_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file. */
void Grid3D::Write_Header_HDF5(hid_t file_id)
{
  hid_t attribute_id, dataspace_id;
  herr_t status;
  hsize_t attr_dims;
  int int_data[3];
  Real Real_data[3];

  // Single attributes first
  attr_dims = 1;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  // Create a group attribute
  attribute_id = H5Acreate(file_id, "gamma", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &gama);
  // Close the attribute
  status = H5Aclose(attribute_id);

  // String attributes
  hid_t stringType = H5Tcopy(H5T_C_S1);
  H5Tset_size(stringType, H5T_VARIABLE);

  attribute_id        = H5Acreate(file_id, "Git Commit Hash", stringType, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  const char *gitHash = GIT_HASH;
  status              = H5Awrite(attribute_id, stringType, &gitHash);
  H5Aclose(attribute_id);

  attribute_id           = H5Acreate(file_id, "Macro Flags", stringType, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  const char *macroFlags = MACRO_FLAGS;
  status                 = H5Awrite(attribute_id, stringType, &macroFlags);
  H5Aclose(attribute_id);

  // attribute to help yt differentiate cholla outputs from outputs produced by other codes
  attribute_id         = H5Acreate(file_id, "cholla", stringType, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  const char *dummyStr = "";  // this doesn't really matter right now
  status               = H5Awrite(attribute_id, stringType, &dummyStr);
  H5Aclose(attribute_id);

  // Numeric Attributes
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &H.t, "t");
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &H.dt, "dt");
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &H.n_step, "n_step");
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &H.n_fields, "n_fields");
  double time_unit     = TIME_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &time_unit, "time_unit");
  double length_unit   = LENGTH_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &length_unit, "length_unit");
  double mass_unit     = MASS_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &mass_unit, "mass_unit");
  double velocity_unit = VELOCITY_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &velocity_unit, "velocity_unit");
  double density_unit  = DENSITY_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &density_unit, "density_unit");
  double energy_unit   = ENERGY_UNIT;
  status               = Write_HDF5_Attribute(file_id, dataspace_id, &energy_unit, "energy_unit");

  #ifdef MHD
  double magnetic_field_unit = MAGNETIC_FIELD_UNIT;
  status                     = Write_HDF5_Attribute(file_id, dataspace_id, &magnetic_field_unit, "magnetic_field_unit");
  #endif  // MHD

  #ifdef COSMOLOGY
  status = Write_HDF5_Attribute(file_id, dataspace_id, &Cosmo.H0, "H0");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &Cosmo.Omega_M, "Omega_M");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &Cosmo.Omega_L, "Omega_L");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &Cosmo.current_z, "Current_z");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &Cosmo.current_a, "Current_a");
  #endif

  // Close the dataspace
  status = H5Sclose(dataspace_id);

  // Now 3D attributes
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  #ifndef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;
  #endif
  #ifdef MPI_CHOLLA
  int_data[0] = nx_global;
  int_data[1] = ny_global;
  int_data[2] = nz_global;
  #endif

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims");

  #ifdef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims_local");

  int_data[0] = nx_local_start;
  int_data[1] = ny_local_start;
  int_data[2] = nz_local_start;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "offset");

  int_data[0] = nproc_x;
  int_data[1] = nproc_y;
  int_data[2] = nproc_z;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "nprocs");
  #endif

  Real_data[0] = H.xbound;
  Real_data[1] = H.ybound;
  Real_data[2] = H.zbound;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "bounds");

  Real_data[0] = H.xdglobal;
  Real_data[1] = H.ydglobal;
  Real_data[2] = H.zdglobal;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "domain");

  Real_data[0] = H.dx;
  Real_data[1] = H.dy;
  Real_data[2] = H.dz;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "dx");

  // Close the dataspace
  status = H5Sclose(dataspace_id);
}

/*! \fn void Write_Header_Rotated_HDF5(hid_t file_id)
 *  \brief Write the relevant header info to the HDF5 file for rotated
 * projection. */
void Grid3D::Write_Header_Rotated_HDF5(hid_t file_id)
{
  hid_t attribute_id, dataspace_id;
  herr_t status;
  hsize_t attr_dims;
  int int_data[3];
  Real Real_data[3];
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
        Get_Position(H.n_ghost + i * (H.nx - 2 * H.n_ghost), H.n_ghost + j * (H.ny - 2 * H.n_ghost),
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

  // Single attributes first
  attr_dims = 1;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);
  // Create a group attribute
  attribute_id = H5Acreate(file_id, "gamma", H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  // Write the attribute data
  status = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, &gama);
  // Close the attribute
  status = H5Aclose(attribute_id);

  // String attributes
  hid_t stringType = H5Tcopy(H5T_C_S1);
  H5Tset_size(stringType, H5T_VARIABLE);

  attribute_id        = H5Acreate(file_id, "Git Commit Hash", stringType, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  const char *gitHash = GIT_HASH;
  status              = H5Awrite(attribute_id, stringType, &gitHash);
  H5Aclose(attribute_id);

  attribute_id           = H5Acreate(file_id, "Macro Flags", stringType, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  const char *macroFlags = MACRO_FLAGS;
  status                 = H5Awrite(attribute_id, stringType, &macroFlags);
  H5Aclose(attribute_id);

  // Numeric Attributes
  status = Write_HDF5_Attribute(file_id, dataspace_id, &H.t, "t");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &H.dt, "dt");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &H.n_step, "n_step");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &H.n_fields, "n_fields");

  // Rotation data
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nx, "nxr");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nz, "nzr");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nx_min, "nx_min");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nz_min, "nz_min");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nx_max, "nx_max");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.nz_max, "nz_max");
  delta  = 180. * R.delta / M_PI;
  status = Write_HDF5_Attribute(file_id, dataspace_id, &delta, "delta");
  theta  = 180. * R.theta / M_PI;
  status = Write_HDF5_Attribute(file_id, dataspace_id, &theta, "theta");
  phi    = 180. * R.phi / M_PI;
  status = Write_HDF5_Attribute(file_id, dataspace_id, &phi, "phi");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.Lx, "Lx");
  status = Write_HDF5_Attribute(file_id, dataspace_id, &R.Lz, "Lz");
  // Close the dataspace
  status = H5Sclose(dataspace_id);

  // Now 3D attributes
  attr_dims = 3;
  // Create the data space for the attribute
  dataspace_id = H5Screate_simple(1, &attr_dims, NULL);

  #ifndef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;
  #endif
  #ifdef MPI_CHOLLA
  int_data[0] = nx_global;
  int_data[1] = ny_global;
  int_data[2] = nz_global;
  #endif

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims");

  #ifdef MPI_CHOLLA
  int_data[0] = H.nx_real;
  int_data[1] = H.ny_real;
  int_data[2] = H.nz_real;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "dims_local");

  int_data[0] = nx_local_start;
  int_data[1] = ny_local_start;
  int_data[2] = nz_local_start;

  status = Write_HDF5_Attribute(file_id, dataspace_id, int_data, "offset");
  #endif

  Real_data[0] = H.xbound;
  Real_data[1] = H.ybound;
  Real_data[2] = H.zbound;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "bounds");

  Real_data[0] = H.xdglobal;
  Real_data[1] = H.ydglobal;
  Real_data[2] = H.zdglobal;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "domain");

  Real_data[0] = H.dx;
  Real_data[1] = H.dy;
  Real_data[2] = H.dz;

  status = Write_HDF5_Attribute(file_id, dataspace_id, Real_data, "dx");

  // Close the dataspace
  status = H5Sclose(dataspace_id);

  chprintf(
      "Outputting rotation data with delta = %e, theta = %e, phi = %e, Lx = "
      "%f, Lz = %f\n",
      R.delta, R.theta, R.phi, R.Lx, R.Lz);
}
#endif  // HDF5

/*! \fn void Write_Grid_Text(FILE *fp)
 *  \brief Write the conserved quantities to a text output file. */
void Grid3D::Write_Grid_Text(FILE *fp)
{
  int id, i, j, k;

  // Write the conserved quantities to the output file

  // 1D case
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    fprintf(fp, "id\trho\tmx\tmy\tmz\tE");
#ifdef MHD
    fprintf(fp, "\tmagX\tmagY\tmagZ");
#endif  // MHD
#ifdef DE
    fprintf(fp, "\tge");
#endif
    fprintf(fp, "\n");
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      id = i;
      fprintf(fp, "%d\t%f\t%f\t%f\t%f\t%f", i - H.n_ghost, C.density[id], C.momentum_x[id], C.momentum_y[id],
              C.momentum_z[id], C.Energy[id]);
#ifdef MHD
      fprintf(fp, "\t%f\t%f\t%f", C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
#endif  // MHD
#ifdef DE
      fprintf(fp, "\t%f", C.GasEnergy[id]);
#endif  // DE
      fprintf(fp, "\n");
    }
#ifdef MHD
    // Save the last line of magnetic fields
    id = H.nx - H.n_ghost;
    fprintf(fp, "%d\tNan\tNan\tNan\tNan\tNan\t%f\t%f\t%f", id, C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
  #ifdef DE
    fprintf(fp, "\tNan");
  #endif  // DE
    fprintf(fp, "\n");
#endif  // MHD
  }

  // 2D case
  else if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    fprintf(fp, "idx\tidy\trho\tmx\tmy\tmz\tE");
#ifdef MHD
    fprintf(fp, "\tmagX\tmagY\tmagZ");
#endif  // MHD
#ifdef DE
    fprintf(fp, "\tge");
#endif
    fprintf(fp, "\n");
    for (i = H.n_ghost; i < H.nx - H.n_ghost; i++) {
      for (j = H.n_ghost; j < H.ny - H.n_ghost; j++) {
        id = i + j * H.nx;
        fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f", i - H.n_ghost, j - H.n_ghost, C.density[id], C.momentum_x[id],
                C.momentum_y[id], C.momentum_z[id], C.Energy[id]);
#ifdef MHD
        fprintf(fp, "\t%f\t%f\t%f", C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
#endif  // MHD
#ifdef DE
        fprintf(fp, "\t%f", C.GasEnergy[id]);
#endif  // DE
        fprintf(fp, "\n");
      }
#ifdef MHD
      // Save the last line of magnetic fields
      id = i + (H.ny - H.n_ghost) * H.nx;
      fprintf(fp, "%d\t%d\tNan\tNan\tNan\tNan\tNan\t%f\t%f\t%f", i - H.n_ghost, H.ny - 2 * H.n_ghost, C.magnetic_x[id],
              C.magnetic_y[id], C.magnetic_z[id]);
  #ifdef DE
      fprintf(fp, "\tNan");
  #endif  // DE
      fprintf(fp, "\n");
#endif  // MHD
    }
#ifdef MHD
    // Save the last line of magnetic fields
    id = H.nx - H.n_ghost + (H.ny - H.n_ghost) * H.nx;
    fprintf(fp, "%d\t%d\tNan\tNan\tNan\tNan\tNan\t%f\t%f\t%f", H.nx - 2 * H.n_ghost, H.ny - 2 * H.n_ghost,
            C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
  #ifdef DE
    fprintf(fp, "\tNan");
  #endif  // DE
    fprintf(fp, "\n");
#endif  // MHD
  }

  // 3D case
  else {
    fprintf(fp, "idx\tidy\tidz\trho\tmx\tmy\tmz\tE");
#ifdef DE
    fprintf(fp, "\tge");
#endif
#ifdef MHD
    fprintf(fp, "\tmagX\tmagY\tmagZ");
#endif  // MHD
    fprintf(fp, "\n");
    for (i = H.n_ghost - 1; i < H.nx - H.n_ghost; i++) {
      for (j = H.n_ghost - 1; j < H.ny - H.n_ghost; j++) {
        for (k = H.n_ghost - 1; k < H.nz - H.n_ghost; k++) {
          id = i + j * H.nx + k * H.nx * H.ny;

          // Exclude the rightmost ghost cell on the "left" side for the hydro
          // variables
          if ((i >= H.n_ghost) and (j >= H.n_ghost) and (k >= H.n_ghost)) {
            fprintf(fp, "%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f", i - H.n_ghost, j - H.n_ghost, k - H.n_ghost, C.density[id],
                    C.momentum_x[id], C.momentum_y[id], C.momentum_z[id], C.Energy[id]);
#ifdef DE
            fprintf(fp, "\t%f", C.GasEnergy[id]);
#endif  // DE
          } else {
            fprintf(fp, "%d\t%d\t%d\tn/a\tn/a\tn/a\tn/a\tn/a", i - H.n_ghost, j - H.n_ghost, k - H.n_ghost);
#ifdef DE
            fprintf(fp, "\tn/a");
#endif  // DE
          }
#ifdef MHD
          fprintf(fp, "\t%f\t%f\t%f", C.magnetic_x[id], C.magnetic_y[id], C.magnetic_z[id]);
#endif  // MHD
          fprintf(fp, "\n");
        }
      }
    }
  }
}

/*! \fn void Write_Grid_Binary(FILE *fp)
 *  \brief Write the conserved quantities to a binary output file. */
void Grid3D::Write_Grid_Binary(FILE *fp)
{
  int id, i, j, k;

  // Write the conserved quantities to the output file

  // 1D case
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    id = H.n_ghost;

    fwrite(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    fwrite(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
#ifdef DE
    fwrite(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
#endif  // DE
  }

  // 2D case
  else if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
    }
#ifdef DE
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fwrite(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
    }
#endif  // DE

  }

  // 3D case
  else {
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.density[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
      }
    }
#ifdef DE
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fwrite(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
      }
    }
#endif  // DE
  }
}

#ifdef HDF5
herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, double *attribute, const char *name)
{
  hid_t attribute_id = H5Acreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_DOUBLE, attribute);
  status             = H5Aclose(attribute_id);
  return status;
}

herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, int *attribute, const char *name)
{
  hid_t attribute_id = H5Acreate(file_id, name, H5T_STD_I32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status      = H5Awrite(attribute_id, H5T_NATIVE_INT, attribute);
  status             = H5Aclose(attribute_id);
  return status;
}

herr_t Read_HDF5_Dataset(hid_t file_id, double *dataset_buffer, const char *name)
{
  hid_t dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  herr_t status    = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status           = H5Dclose(dataset_id);
  return status;
}

herr_t Read_HDF5_Dataset(hid_t file_id, float *dataset_buffer, const char *name)
{
  hid_t dataset_id = H5Dopen(file_id, name, H5P_DEFAULT);
  herr_t status    = H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  status           = H5Dclose(dataset_id);
  return status;
}

// Helper function which uses the correct HDF5 arguments based on the type of
// dataset_buffer to avoid writing garbage
herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, double *dataset_buffer, const char *name)
{
  // Create the dataset id
  hid_t dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F64BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the array to file
  herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // Free the dataset id
  status = H5Dclose(dataset_id);
  return status;
}

herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, float *dataset_buffer, const char *name)
{
  // Create the dataset id
  hid_t dataset_id = H5Dcreate(file_id, name, H5T_IEEE_F32BE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // Write the array to file
  herr_t status = H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
  // Free the dataset id
  status = H5Dclose(dataset_id);
  return status;
}

void Write_HDF5_Field_1D_CPU(Header H, hid_t file_id, hid_t dataspace_id, Real *dataset_buffer, Real *source,
                             const char *name)
{
  // Copy non-ghost source to Buffer
  int id = H.n_ghost;
  memcpy(&dataset_buffer[0], &(source[id]), H.nx_real * sizeof(Real));
  // Buffer write to HDF5 Dataset
  herr_t status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer, name);
}

void Write_HDF5_Field_1D_CPU(Header H, hid_t file_id, hid_t dataspace_id, float *dataset_buffer, double *source,
                             const char *name)
{
  // Copy non-ghost source to Buffer with conversion from double to float
  int i;
  for (i = 0; i < H.nx_real; i++) {
    dataset_buffer[i] = (float)source[i + H.n_ghost];
  }
  // Buffer write to HDF5 Dataset
  herr_t status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer, name);
}

void Write_HDF5_Field_2D_CPU(Header H, hid_t file_id, hid_t dataspace_id, Real *dataset_buffer, Real *source,
                             const char *name)
{
  int i, j, id, buf_id;
  // Copy non-ghost source to Buffer
  for (j = 0; j < H.ny_real; j++) {
    for (i = 0; i < H.nx_real; i++) {
      id                     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx;
      buf_id                 = j + i * H.ny_real;
      dataset_buffer[buf_id] = source[id];
    }
  }
  // Buffer write to HDF5 Dataset
  herr_t status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer, name);
}

// Convert double to float if necessary
void Write_HDF5_Field_2D_CPU(Header H, hid_t file_id, hid_t dataspace_id, float *dataset_buffer, double *source,
                             const char *name)
{
  int i, j, id, buf_id;
  // Copy non-ghost source to Buffer with conversion to float
  for (j = 0; j < H.ny_real; j++) {
    for (i = 0; i < H.nx_real; i++) {
      id                     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx;
      buf_id                 = j + i * H.ny_real;
      dataset_buffer[buf_id] = (float)source[id];
    }
  }
  // Buffer write to HDF5 Dataset
  herr_t status = Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer, name);
}

/* \brief Before HDF5 reads data into a buffer, remap and write grid to HDF5 buffer. */
void Fill_HDF5_Buffer_From_Grid_CPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                    Real *hdf5_buffer, Real *grid_buffer)
{
  int i, j, k, id, buf_id;
  // 3D case
  if (nx > 1 && ny > 1 && nz > 1) {
    for (k = 0; k < nz_real; k++) {
      for (j = 0; j < ny_real; j++) {
        for (i = 0; i < nx_real; i++) {
          id                  = (i + n_ghost) + (j + n_ghost) * nx + (k + n_ghost) * nx * ny;
          buf_id              = k + j * nz_real + i * nz_real * ny_real;
          hdf5_buffer[buf_id] = grid_buffer[id];
        }
      }
    }
    return;
  }

  // 2D case
  if (nx > 1 && ny > 1 && nz == 1) {
    for (j = 0; j < ny_real; j++) {
      for (i = 0; i < nx_real; i++) {
        id                  = (i + n_ghost) + (j + n_ghost) * nx;
        buf_id              = j + i * ny_real;
        hdf5_buffer[buf_id] = grid_buffer[id];
      }
    }
    return;
  }

  // 1D case
  if (nx > 1 && ny == 1 && nz == 1) {
    id = n_ghost;
    memcpy(&hdf5_buffer[0], &grid_buffer[id], nx_real * sizeof(Real));
    return;
  }
}

/* \brief Before HDF5 reads data into a buffer, remap and write grid to HDF5 buffer. */
void Fill_HDF5_Buffer_From_Grid_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                    Real *hdf5_buffer, Real *device_hdf5_buffer, Real *device_grid_buffer);
// From src/io/io_gpu

// Set up dataspace for grid formatted data and write dataset
void Write_HDF5_Dataset_Grid(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, hid_t file_id,
                             Real *dataset_buffer, const char *name)
{
  // Set up dataspace

  hid_t dataspace_id;
  // 1-D Case
  if (nx > 1 && ny == 1 && nz == 1) {
    int rank = 1;
    hsize_t dims[1];
    dims[0]      = nx_real;
    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }
  // 2-D Case
  if (nx > 1 && ny > 1 && nz == 1) {
    int rank = 2;
    hsize_t dims[2];
    dims[0]      = nx_real;
    dims[1]      = ny_real;
    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }
  // 3-D Case
  if (nx > 1 && ny > 1 && nz > 1) {
    int rank = 3;
    hsize_t dims[3];
    dims[0]      = nx_real;
    dims[1]      = ny_real;
    dims[2]      = nz_real;
    dataspace_id = H5Screate_simple(rank, dims, NULL);
  }

  // Write to HDF5 file

  Write_HDF5_Dataset(file_id, dataspace_id, dataset_buffer, name);

  // Close dataspace
  herr_t status = H5Sclose(dataspace_id);
}

// Data moves from host grid_buffer to dataset_buffer to hdf5 file
void Write_Grid_HDF5_Field_CPU(Header H, hid_t file_id, Real *dataset_buffer, Real *grid_buffer, const char *name)
{
  Fill_HDF5_Buffer_From_Grid_CPU(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, H.n_ghost, dataset_buffer,
                                 grid_buffer);
  Write_HDF5_Dataset_Grid(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, file_id, dataset_buffer, name);
}

// Data moves from device_grid_buffer to device_hdf5_buffer to dataset_buffer to hdf5 file
void Write_Grid_HDF5_Field_GPU(Header H, hid_t file_id, Real *dataset_buffer, Real *device_hdf5_buffer,
                               Real *device_grid_buffer, const char *name)
{
  Fill_HDF5_Buffer_From_Grid_GPU(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, H.n_ghost, dataset_buffer,
                                 device_hdf5_buffer, device_grid_buffer);
  Write_HDF5_Dataset_Grid(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, file_id, dataset_buffer, name);
}

void Write_Generic_HDF5_Field_CPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                  hid_t file_id, Real *dataset_buffer, Real *source_buffer, const char *name)
{
  Fill_HDF5_Buffer_From_Grid_CPU(nx, ny, nz, nx_real, ny_real, nz_real, n_ghost, dataset_buffer, source_buffer);
  Write_HDF5_Dataset_Grid(nx, ny, nz, nx_real, ny_real, nz_real, file_id, dataset_buffer, name);
}

void Write_Generic_HDF5_Field_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                  hid_t file_id, Real *dataset_buffer, Real *device_hdf5_buffer, Real *source_buffer,
                                  const char *name)
{
  Fill_HDF5_Buffer_From_Grid_GPU(nx, ny, nz, nx_real, ny_real, nz_real, n_ghost, dataset_buffer, device_hdf5_buffer,
                                 source_buffer);
  Write_HDF5_Dataset_Grid(nx, ny, nz, nx_real, ny_real, nz_real, file_id, dataset_buffer, name);
}

/*! \fn void Write_Grid_HDF5(hid_t file_id)
 *  \brief Write the grid to a file, at the current simulation time. */
void Grid3D::Write_Grid_HDF5(hid_t file_id)
{
  int i, j, k, id, buf_id;
  hid_t dataset_id, dataspace_id;
  hid_t dataset_id_full, dataspace_id_full;
  Real *dataset_buffer;
  herr_t status;

  bool output_energy;
  bool output_momentum;

  #ifdef OUTPUT_ENERGY
  output_energy = true;
  #else   // not OUTPUT_ENERGY
  output_energy = false;
  #endif  // OUTPUT_ENERGY

  #ifdef OUTPUT_MOMENTUM
  output_momentum = true;
  #else   // not OUTPUT_MOMENTUM
  output_momentum = false;
  #endif  // OUTPUT_MOMENTUM

  #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  bool output_metals, output_electrons, output_full_ionization;
    #ifdef OUTPUT_METALS
  output_metals = true;
    #else   // not OUTPUT_METALS
  output_metals = false;
    #endif  // OUTPUT_METALS
    #ifdef OUTPUT_ELECTRONS
  output_electrons = true;
    #else   // not OUTPUT_ELECTRONS
  output_electrons = false;
    #endif  // OUTPUT_ELECTRONS
    #ifdef OUTPUT_FULL_IONIZATION
  output_full_ionization = true;
    #else   // not OUTPUT_FULL_IONIZATION
  output_full_ionization = false;
    #endif  // OUTPUT_FULL_IONIZATION

  #endif  // COOLING_GRACKLE or CHEMISTRY_GPU

  // Allocate necessary buffers
  int nx_dset = H.nx_real;
  int ny_dset = H.ny_real;
  int nz_dset = H.nz_real;
  #ifdef MHD
  size_t buffer_size = (nx_dset + 1) * (ny_dset + 1) * (nz_dset + 1);
  #else
  size_t buffer_size = nx_dset * ny_dset * nz_dset;
  #endif
  cuda_utilities::DeviceVector<Real> static device_dataset_vector{buffer_size};
  dataset_buffer = (Real *)malloc(buffer_size * sizeof(Real));

  // Start writing fields

  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_density, "/density");
  if (output_momentum || H.Output_Complete_Data) {
    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_x, "/momentum_x");
    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_y, "/momentum_y");
    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_momentum_z, "/momentum_z");
  }
  if (output_energy || H.Output_Complete_Data) {
    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_Energy, "/Energy");
  #ifdef DE
    Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_GasEnergy, "/GasEnergy");
  #endif
  }

  #ifdef SCALAR

    #ifdef BASIC_SCALAR
  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_basic_scalar, "/scalar0");
    #endif  // BASIC_SCALAR

    #ifdef DUST
  Write_Grid_HDF5_Field_GPU(H, file_id, dataset_buffer, device_dataset_vector.data(), C.d_dust_density,
                            "/dust_density");
    #endif  // DUST

    #ifdef OUTPUT_CHEMISTRY
      #ifdef CHEMISTRY_GPU
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.HI_density, "/HI_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.HII_density, "/HII_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.HeI_density, "/HeI_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.HeII_density, "/HeII_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.HeIII_density, "/HeIII_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, C.e_density, "/e_density");
      #elif defined(COOLING_GRACKLE)
  // Cool fields are CPU (host) only
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.HI_density, "/HI_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.HII_density, "/HII_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.HeI_density, "/HeI_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.HeII_density, "/HeII_density");
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.HeIII_density, "/HeIII_density");
  if (output_electrons || H.Output_Complete_Data) {
    Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.e_density, "/e_density");
  }
      #endif
    #endif  // OUTPUT_CHEMISTRY

    #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)

      #ifdef GRACKLE_METALS
  if (output_metals || H.Output_Complete_Data) {
    Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.fields.metal_density, "/metal_density");
  }
      #endif  // GRACKLE_METALS

      #ifdef OUTPUT_TEMPERATURE
        #ifdef CHEMISTRY_GPU
  Compute_Gas_Temperature(Chem.Fields.temperature_h, false);
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Chem.Fields.temperature_h, "/temperature");
        #elif defined(COOLING_GRACKLE)
  Write_Grid_HDF5_Field_CPU(H, file_id, dataset_buffer, Cool.temperature, "/temperature");
        #endif
      #endif

    #endif  // COOLING_GRACKLE || CHEMISTRY_GPU

  #endif  // SCALAR

  // 3D case
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
  #if defined(GRAVITY) && defined(OUTPUT_POTENTIAL)
    Write_Generic_HDF5_Field_GPU(Grav.nx_local + 2 * N_GHOST_POTENTIAL, Grav.ny_local + 2 * N_GHOST_POTENTIAL,
                                 Grav.nz_local + 2 * N_GHOST_POTENTIAL, Grav.nx_local, Grav.ny_local, Grav.nz_local,
                                 N_GHOST_POTENTIAL, file_id, dataset_buffer, device_dataset_vector.data(),
                                 Grav.F.potential_d, "/grav_potential");
  #endif  // GRAVITY and OUTPUT_POTENTIAL

  #ifdef MHD
    if (H.Output_Complete_Data) {
      Write_HDF5_Field_3D(H.nx, H.ny, H.nx_real + 1, H.ny_real, H.nz_real, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), C.d_magnetic_x, "/magnetic_x", 0);
      Write_HDF5_Field_3D(H.nx, H.ny, H.nx_real, H.ny_real + 1, H.nz_real, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), C.d_magnetic_y, "/magnetic_y", 1);
      Write_HDF5_Field_3D(H.nx, H.ny, H.nx_real, H.ny_real, H.nz_real + 1, H.n_ghost, file_id, dataset_buffer,
                          device_dataset_vector.data(), C.d_magnetic_z, "/magnetic_z", 2);
    }
  #endif  // MHD
  }

  free(dataset_buffer);
}
#endif  // HDF5

#ifdef HDF5
/*! \fn void Write_Projection_HDF5(hid_t file_id)
 *  \brief Write projected density and temperature data to a file, at the
 * current simulation time. */
void Grid3D::Write_Projection_HDF5(hid_t file_id)
{
  hid_t dataset_id, dataspace_xy_id, dataspace_xz_id;
  Real *dataset_buffer_dxy, *dataset_buffer_dxz;
  Real *dataset_buffer_Txy, *dataset_buffer_Txz;
  herr_t status;
  Real dxy, dxz, Txy, Txz;
  #ifdef DUST
  Real dust_xy, dust_xz;
  Real *dataset_buffer_dust_xy, *dataset_buffer_dust_xz;
  #endif

  Real mu = 0.6;

  // 3D
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    int nx_dset = H.nx_real;
    int ny_dset = H.ny_real;
    int nz_dset = H.nz_real;
    hsize_t dims[2];
    dataset_buffer_dxy = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
    dataset_buffer_dxz = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
    dataset_buffer_Txy = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
    dataset_buffer_Txz = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  #ifdef DUST
    dataset_buffer_dust_xy = (Real *)malloc(H.nx_real * H.ny_real * sizeof(Real));
    dataset_buffer_dust_xz = (Real *)malloc(H.nx_real * H.nz_real * sizeof(Real));
  #endif

    // Create the data space for the datasets
    dims[0]         = nx_dset;
    dims[1]         = ny_dset;
    dataspace_xy_id = H5Screate_simple(2, dims, NULL);
    dims[1]         = nz_dset;
    dataspace_xz_id = H5Screate_simple(2, dims, NULL);

    // Copy the xy density and temperature projections to the memory buffer
    for (int j = 0; j < H.ny_real; j++) {
      for (int i = 0; i < H.nx_real; i++) {
        dxy = 0;
        Txy = 0;
  #ifdef DUST
        dust_xy = 0;
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
          auto const [magnetic_x, magnetic_y, magnetic_z] =
              mhd::utils::cellCenteredMagneticFields(C.host, id, xid, yid, zid, H.n_cells, H.nx, H.ny);
    #else   // MHD is not defined
          Real const magnetic_x = 0.0, magnetic_y = 0.0, magnetic_z = 0.0;
    #endif  // MHD

          Real const T =
              hydro_utilities::Calc_Temp_Conserved(E, d, mx, my, mz, gama, n, magnetic_x, magnetic_y, magnetic_z);
  #endif    // DE

          Txy += T * d * H.dz;
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
        dxz = 0;
        Txz = 0;
  #ifdef DUST
        dust_xz = 0;
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
          // calculate number density
          Real const n = d * DENSITY_UNIT / (mu * MP);
  #ifdef DE
          Real const T = hydro_utilities::Calc_Temp_DE(C.GasEnergy[id], gama, n);
  #else  // DE is not defined
          Real const mx = C.momentum_x[id];
          Real const my = C.momentum_y[id];
          Real const mz = C.momentum_z[id];
          Real const E  = C.Energy[id];

    #ifdef MHD
          auto const [magnetic_x, magnetic_y, magnetic_z] =
              mhd::utils::cellCenteredMagneticFields(C.host, id, xid, yid, zid, H.n_cells, H.nx, H.ny);
    #else   // MHD is not defined
          Real const magnetic_x = 0.0, magnetic_y = 0.0, magnetic_z = 0.0;
    #endif  // MHD

          Real const T =
              hydro_utilities::Calc_Temp_Conserved(E, d, mx, my, mz, gama, n, magnetic_x, magnetic_y, magnetic_z);
  #endif    // DE
          Txz += T * d * H.dy;
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
    status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_dxy, "/d_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_dxz, "/d_xz");
    status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_Txy, "/T_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_Txz, "/T_xz");
  #ifdef DUST
    status = Write_HDF5_Dataset(file_id, dataspace_xy_id, dataset_buffer_dust_xy, "/d_dust_xy");
    status = Write_HDF5_Dataset(file_id, dataspace_xz_id, dataset_buffer_dust_xz, "/d_dust_xz");
  #endif

    // Free the dataspace ids
    status = H5Sclose(dataspace_xz_id);
    status = H5Sclose(dataspace_xy_id);
  } else {
    printf("Projection write only works for 3D data.\n");
  }

  free(dataset_buffer_dxy);
  free(dataset_buffer_dxz);
  free(dataset_buffer_Txy);
  free(dataset_buffer_Txz);
  #ifdef DUST
  free(dataset_buffer_dust_xy);
  free(dataset_buffer_dust_xz);
  #endif  // DUST
}
#endif  // HDF5

#ifdef HDF5
/*! \fn void Write_Rotated_Projection_HDF5(hid_t file_id)
 *  \brief Write rotated projected data to a file, at the current simulation
 * time. */
void Grid3D::Write_Rotated_Projection_HDF5(hid_t file_id)
{
  hid_t dataset_id, dataspace_xzr_id;
  Real *dataset_buffer_dxzr;
  Real *dataset_buffer_Txzr;
  Real *dataset_buffer_vxxzr;
  Real *dataset_buffer_vyxzr;
  Real *dataset_buffer_vzxzr;

  herr_t status;
  Real dxy, dxz, Txy, Txz;
  Real d, n, vx, vy, vz;

  Real x, y, z;      // cell positions
  Real xp, yp, zp;   // rotated positions
  Real alpha, beta;  // projected positions
  int ix, iz;        // projected index positions

  n       = 0;
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
          Get_Position(i + H.n_ghost, j + H.n_ghost, k + H.n_ghost, &x, &y, &z);

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
            n = d * DENSITY_UNIT / (mu * MP);

  // calculate temperature
  #ifdef DE
            Real const T = hydro_utilities::Calc_Temp_DE(C.GasEnergy[id], gama, n);
  #else  // DE is not defined
            Real const mx = C.momentum_x[id];
            Real const my = C.momentum_y[id];
            Real const mz = C.momentum_z[id];
            Real const E  = C.Energy[id];

    #ifdef MHD
            auto const [magnetic_x, magnetic_y, magnetic_z] =
                mhd::utils::cellCenteredMagneticFields(C.host, id, xid, yid, zid, H.n_cells, H.nx, H.ny);
    #else   // MHD is not defined
            Real const magnetic_x = 0.0, magnetic_y = 0.0, magnetic_z = 0.0;
    #endif  // MHD

            Real const T =
                hydro_utilities::Calc_Temp_Conserved(E, d, mx, my, mz, gama, n, magnetic_x, magnetic_y, magnetic_z);
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
}
#endif  // HDF5

#ifdef HDF5
/*! \fn void Write_Slices_HDF5(hid_t file_id)
 *  \brief Write centered xy, xz, and yz slices of all variables to a file,
     at the current simulation time. */
void Grid3D::Write_Slices_HDF5(hid_t file_id)
{
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

/*! \fn void Read_Grid(struct Parameters P)
 *  \brief Read in grid data from an output file. */
void Grid3D::Read_Grid(struct Parameters P)
{
  ScopedTimer timer("Read_Grid");
  int nfile = P.nfile;  // output step you want to read from

  // create the filename to read from
  // assumes your data is in the outdir specified in the input file
  // strcpy(filename, P.outdir);
  // Changed to read initial conditions from indir
  std::string filename(P.indir);
  filename += std::to_string(P.nfile);

#if defined BINARY
  filename += ".bin";
#elif defined HDF5
  filename += ".h5";
#endif  // BINARY or HDF5
// for now assumes you will run on the same number of processors
#ifdef MPI_CHOLLA
  #ifdef TILED_INITIAL_CONDITIONS
  sprintf(filename, "%sics_%dMpc_%d.h5", P.indir, (int)P.tile_length / 1000,
          H.nx_real);  // Everyone reads the same file
  #else                // TILED_INITIAL_CONDITIONS is not defined
  filename += "." + std::to_string(procID);
  #endif               // TILED_INITIAL_CONDITIONS
#endif                 // MPI_CHOLLA

#if defined BINARY
  FILE *fp;
  // open the file
  fp = fopen(filename, "r");
  if (!fp) {
    printf("Unable to open input file.\n");
    exit(0);
  }

  // read in grid data
  Read_Grid_Binary(fp);

  // close the file
  fclose(fp);

#elif defined HDF5
  hid_t file_id;
  herr_t status;

  // open the file
  file_id = H5Fopen(filename.data(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    std::cout << "Unable to open input file: " << filename << std::endl;
    exit(0);
  }

  // read in grid data
  Read_Grid_HDF5(file_id, P);

  // close the file
  status = H5Fclose(file_id);
#endif  // BINARY or HDF5
}

/*! \fn Read_Grid_Binary(FILE *fp)
 *  \brief Read in grid data from a binary file. */
void Grid3D::Read_Grid_Binary(FILE *fp)
{
  int id, i, j, k;
  size_t rs;

  // Read in the header data
  rs = fread(&H.n_cells, sizeof(int), 1, fp);
  rs = fread(&H.n_ghost, sizeof(int), 1, fp);
  rs = fread(&H.nx, sizeof(int), 1, fp);
  rs = fread(&H.ny, sizeof(int), 1, fp);
  rs = fread(&H.nz, sizeof(int), 1, fp);
  rs = fread(&H.nx_real, sizeof(int), 1, fp);
  rs = fread(&H.ny_real, sizeof(int), 1, fp);
  rs = fread(&H.nz_real, sizeof(int), 1, fp);
  rs = fread(&H.xbound, sizeof(Real), 1, fp);
  rs = fread(&H.ybound, sizeof(Real), 1, fp);
  rs = fread(&H.zbound, sizeof(Real), 1, fp);
  rs = fread(&H.xblocal, sizeof(Real), 1, fp);
  rs = fread(&H.yblocal, sizeof(Real), 1, fp);
  rs = fread(&H.zblocal, sizeof(Real), 1, fp);
  rs = fread(&H.xdglobal, sizeof(Real), 1, fp);
  rs = fread(&H.ydglobal, sizeof(Real), 1, fp);
  rs = fread(&H.zdglobal, sizeof(Real), 1, fp);
  rs = fread(&H.dx, sizeof(Real), 1, fp);
  rs = fread(&H.dy, sizeof(Real), 1, fp);
  rs = fread(&H.dz, sizeof(Real), 1, fp);
  rs = fread(&H.t, sizeof(Real), 1, fp);
  rs = fread(&H.dt, sizeof(Real), 1, fp);
  rs = fread(&H.t_wall, sizeof(Real), 1, fp);
  rs = fread(&H.n_step, sizeof(int), 1, fp);

// Read in the conserved quantities from the input file
#ifdef WITH_GHOST
  fread(&(C.density[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_x[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_y[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.momentum_z[id]), sizeof(Real), H.n_cells, fp);
  fread(&(C.Energy[id]), sizeof(Real), H.n_cells, fp);
#endif  // WITH_GHOST

#ifdef NO_GHOST
  // 1D case
  if (H.nx > 1 && H.ny == 1 && H.nz == 1) {
    id = H.n_ghost;

    fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
  #ifdef DE
    fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
  #endif
  }

  // 2D case
  else if (H.nx > 1 && H.ny > 1 && H.nz == 1) {
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
    }
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
    }
  #ifdef DE
    for (j = 0; j < H.ny_real; j++) {
      id = H.n_ghost + (j + H.n_ghost) * H.nx;
      fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
    }
  #endif
  }

  // 3D case
  else {
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.density[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.momentum_x[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.momentum_y[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.momentum_z[id]), sizeof(Real), H.nx_real, fp);
      }
    }
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.Energy[id]), sizeof(Real), H.nx_real, fp);
      }
    }
  #ifdef DE
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        id = H.n_ghost + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        fread(&(C.GasEnergy[id]), sizeof(Real), H.nx_real, fp);
      }
    }
  #endif
  }
#endif
}

#ifdef HDF5

/* \brief After HDF5 reads data into a buffer, remap and write to grid buffer. */
void Fill_Grid_From_HDF5_Buffer(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                Real *hdf5_buffer, Real *grid_buffer)
{
  // Note: for 1D ny_real and nz_real are not used
  // And for 2D nz_real is not used.
  // This protects the magnetic case where ny_real/nz_real += 1

  int i, j, k, id, buf_id;
  // 3D case
  if (nx > 1 && ny > 1 && nz > 1) {
    for (k = 0; k < nz_real; k++) {
      for (j = 0; j < ny_real; j++) {
        for (i = 0; i < nx_real; i++) {
          id              = (i + n_ghost) + (j + n_ghost) * nx + (k + n_ghost) * nx * ny;
          buf_id          = k + j * nz_real + i * nz_real * ny_real;
          grid_buffer[id] = hdf5_buffer[buf_id];
        }
      }
    }
    return;
  }

  // 2D case
  if (nx > 1 && ny > 1 && nz == 1) {
    for (j = 0; j < ny_real; j++) {
      for (i = 0; i < nx_real; i++) {
        id              = (i + n_ghost) + (j + n_ghost) * nx;
        buf_id          = j + i * ny_real;
        grid_buffer[id] = hdf5_buffer[buf_id];
      }
    }
    return;
  }

  // 1D case
  if (nx > 1 && ny == 1 && nz == 1) {
    id = n_ghost;
    memcpy(&grid_buffer[id], &hdf5_buffer[0], nx_real * sizeof(Real));
    return;
  }
}

void Read_Grid_HDF5_Field(hid_t file_id, Real *dataset_buffer, Header H, Real *grid_buffer, const char *name)
{
  Read_HDF5_Dataset(file_id, dataset_buffer, name);
  Fill_Grid_From_HDF5_Buffer(H.nx, H.ny, H.nz, H.nx_real, H.ny_real, H.nz_real, H.n_ghost, dataset_buffer, grid_buffer);
}

void Read_Grid_HDF5_Field_Magnetic(hid_t file_id, Real *dataset_buffer, Header H, Real *grid_buffer, const char *name)
{
  // Magnetic has 1 more real cell, 1 fewer n_ghost on one side.
  Read_HDF5_Dataset(file_id, dataset_buffer, name);
  Fill_Grid_From_HDF5_Buffer(H.nx, H.ny, H.nz, H.nx_real + 1, H.ny_real + 1, H.nz_real + 1, H.n_ghost - 1,
                             dataset_buffer, grid_buffer);
}

/*! \fn void Read_Grid_HDF5(hid_t file_id)
 *  \brief Read in grid data from an hdf5 file. */
void Grid3D::Read_Grid_HDF5(hid_t file_id, struct Parameters P)
{
  int i, j, k, id, buf_id;
  hid_t attribute_id, dataset_id;
  Real *dataset_buffer;
  herr_t status;

  // Read in header values not set by grid initialization
  attribute_id = H5Aopen(file_id, "gamma", H5P_DEFAULT);
  status       = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &gama);
  status       = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "t", H5P_DEFAULT);
  status       = H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &H.t);
  status       = H5Aclose(attribute_id);
  attribute_id = H5Aopen(file_id, "n_step", H5P_DEFAULT);
  status       = H5Aread(attribute_id, H5T_NATIVE_INT, &H.n_step);
  status       = H5Aclose(attribute_id);

  #ifdef MHD
  dataset_buffer = (Real *)malloc((H.nz_real + 1) * (H.ny_real + 1) * (H.nx_real + 1) * sizeof(Real));
  #else
  dataset_buffer = (Real *)malloc((H.nz_real) * (H.ny_real) * (H.nx_real) * sizeof(Real));
  #endif

  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.density, "/density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.momentum_x, "/momentum_x");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.momentum_y, "/momentum_y");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.momentum_z, "/momentum_z");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.Energy, "/Energy");
  #ifdef DE
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.GasEnergy, "/GasEnergy");
  #endif

  #ifdef SCALAR

    #ifdef BASIC_SCALAR
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.scalar, "/scalar0");
    #endif  // BASIC_SCALAR

    #ifdef DUST
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.dust_density, "/dust_density");
    #endif  // DUST

    #if defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.HI_density, "/HI_density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.HII_density, "/HII_density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.HeI_density, "/HeI_density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.HeII_density, "/HeII_density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.HeIII_density, "/HeIII_density");
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.e_density, "/e_density");
      #ifdef GRACKLE_METALS
  Read_Grid_HDF5_Field(file_id, dataset_buffer, H, C.metal_density, "/metal_density");
      #endif  // GRACKLE_METALS
    #endif    // COOLING_GRACKLE , CHEMISTRY_GPU

  #endif  // SCALAR

  // MHD only valid in 3D case
  if (H.nx > 1 && H.ny > 1 && H.nz > 1) {
    // Compute Statistic of Initial data
    Real mean_l, min_l, max_l;
    Real mean_g, min_g, max_g;

  #ifdef MHD
    // Open the x magnetic field dataset
    dataset_id = H5Dopen(file_id, "/magnetic_x", H5P_DEFAULT);
    // Read the x magnetic field array into the dataset buffer  // NOTE: NEED TO
    // FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l  = 1e65;
    max_l  = -1;
    // Copy the x magnetic field array to the grid
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real; j++) {
        for (i = 0; i < H.nx_real + 1; i++) {
          id               = (i + H.n_ghost - 1) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
          buf_id           = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
          C.magnetic_x[id] = dataset_buffer[buf_id];

          mean_l += std::abs(C.magnetic_x[id]);
          max_l = std::max(max_l, std::abs(C.magnetic_x[id]));
          min_l = std::min(min_l, std::abs(C.magnetic_x[id]));
        }
      }
    }
    mean_l /= ((H.nz_real + 1) * (H.ny_real) * (H.nx_real));

    #if MPI_CHOLLA
    mean_g = ReduceRealAvg(mean_l);
    max_g  = ReduceRealMax(max_l);
    min_g  = ReduceRealMin(min_l);
    mean_l = mean_g;
    max_l  = max_g;
    min_l  = min_g;
    #endif  // MPI_CHOLLA

    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
    chprintf(
        " abs(Magnetic X)  Mean: %f   Min: %f   Max: %f      [ Msun^1/2 "
        "kpc^-1/2 s^-1] \n",
        mean_l, min_l, max_l);
    #endif  // PRINT_INITIAL_STATS and COSMOLOGY

    // Open the y magnetic field dataset
    dataset_id = H5Dopen(file_id, "/magnetic_y", H5P_DEFAULT);
    // Read the y magnetic field array into the dataset buffer  // NOTE: NEED TO
    // FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l  = 1e65;
    max_l  = -1;
    // Copy the y magnetic field array to the grid
    for (k = 0; k < H.nz_real; k++) {
      for (j = 0; j < H.ny_real + 1; j++) {
        for (i = 0; i < H.nx_real; i++) {
          id               = (i + H.n_ghost) + (j + H.n_ghost - 1) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
          buf_id           = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real + 1);
          C.magnetic_y[id] = dataset_buffer[buf_id];

          mean_l += std::abs(C.magnetic_x[id]);
          max_l = std::max(max_l, std::abs(C.magnetic_x[id]));
          min_l = std::min(min_l, std::abs(C.magnetic_x[id]));
        }
      }
    }
    mean_l /= ((H.nz_real) * (H.ny_real + 1) * (H.nx_real));

    #if MPI_CHOLLA
    mean_g = ReduceRealAvg(mean_l);
    max_g  = ReduceRealMax(max_l);
    min_g  = ReduceRealMin(min_l);
    mean_l = mean_g;
    max_l  = max_g;
    min_l  = min_g;
    #endif  // MPI_CHOLLA

    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
    chprintf(
        " abs(Magnetic Y)  Mean: %f   Min: %f   Max: %f      [ Msun^1/2 "
        "kpc^-1/2 s^-1] \n",
        mean_l, min_l, max_l);
    #endif  // PRINT_INITIAL_STATS and COSMOLOGY

    // Open the z magnetic field dataset
    dataset_id = H5Dopen(file_id, "/magnetic_z", H5P_DEFAULT);
    // Read the z magnetic field array into the dataset buffer  // NOTE: NEED TO
    // FIX FOR FLOAT REAL!!!
    status = H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataset_buffer);
    // Free the dataset id
    status = H5Dclose(dataset_id);

    mean_l = 0;
    min_l  = 1e65;
    max_l  = -1;
    // Copy the z magnetic field array to the grid
    for (k = 0; k < H.nz_real + 1; k++) {
      for (j = 0; j < H.ny_real; j++) {
        for (i = 0; i < H.nx_real; i++) {
          id               = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost - 1) * H.nx * H.ny;
          buf_id           = k + j * (H.nz_real + 1) + i * (H.nz_real + 1) * (H.ny_real);
          C.magnetic_z[id] = dataset_buffer[buf_id];

          mean_l += std::abs(C.magnetic_x[id]);
          max_l = std::max(max_l, std::abs(C.magnetic_x[id]));
          min_l = std::min(min_l, std::abs(C.magnetic_x[id]));
        }
      }
    }
    mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real + 1));

    #if MPI_CHOLLA
    mean_g = ReduceRealAvg(mean_l);
    max_g  = ReduceRealMax(max_l);
    min_g  = ReduceRealMin(min_l);
    mean_l = mean_g;
    max_l  = max_g;
    min_l  = min_g;
    #endif  // MPI_CHOLLA

    #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
    chprintf(
        " abs(Magnetic Z)  Mean: %f   Min: %f   Max: %f      [ Msun^1/2 "
        "kpc^-1/2 s^-1] \n",
        mean_l, min_l, max_l);
    #endif  // PRINT_INITIAL_STATS and COSMOLOGY
  #endif    // MHD
  }
  free(dataset_buffer);
}
#endif

/* MPI-safe printf routine */
int chprintf(const char *__restrict sdata, ...)  // NOLINT(cert-dcl50-cpp)
{
  int code = 0;
  /*limit printf to root process only*/
  if (Is_Root_Proc()) {
    va_list ap;
    va_start(ap, sdata);
    code = vfprintf(stdout, sdata, ap);  // NOLINT(clang-analyzer-valist.Uninitialized)
    va_end(ap);
    fflush(stdout);
  }

  return code;
}

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

void Write_Debug(Real *Value, const char *fname, int nValues, int iProc)
{
  char fn[1024];
  int ret;

  sprintf(fn, "%s_%07d.txt", fname, iProc);
  FILE *fp = fopen(fn, "w");

  for (int iV = 0; iV < nValues; iV++) {
    fprintf(fp, "%e\n", Value[iV]);
  }

  fclose(fp);
}

void Ensure_Outdir_Exists(std::string outdir)
{
  if (outdir == "") {
    return;
  } else if (Is_Root_Proc()) {
    // if the last character of outdir is not a '/', then the substring of
    // characters after the final '/' (or entire string if there isn't any '/')
    // is treated as a file-prefix
    //
    // this is accomplished here:
    std::filesystem::path without_file_prefix = std::filesystem::path(outdir).parent_path();

    if (!without_file_prefix.empty()) {
      // try to create all directories specified within outdir (does nothing if
      // the directories already exist)
      std::error_code err_code;
      std::filesystem::create_directories(without_file_prefix, err_code);

      // confirm that an error-code wasn't set & that the path actually refers
      // to a directory (it's unclear from docs whether err-code is set in that
      // case)
      if (err_code or not std::filesystem::is_directory(without_file_prefix)) {
        CHOLLA_ERROR(
            "something went wrong while trying to create the path to the "
            "output-dir: %s",
            outdir.c_str());
      }
    }
  }

  // this barrier ensures we won't ever encounter a scenario when 1 process
  // tries to write a file to a non-existent directory before the root process
  // has a chance to create it
#ifdef MPI_CHOLLA
  MPI_Barrier(world);
#endif
}
