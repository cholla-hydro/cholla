/*! \file
 *  \brief Implements logic pertaining to reading and writing data.
 */

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
#include <unordered_set>
#ifdef HDF5
  #include <hdf5.h>
#endif  // HDF5
#include "../grid/grid3D.h"
#include "../io/WriterManager.h"
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
#ifdef RT
  #include "../radiation/radiation.h"
#endif

/* Generate the log output file */
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

/* Write an entry in the log output file */
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
void Write_Data(Grid3D &G, struct Parameters P, int nfile, const io::WriterManager &write_manager)
{
  cudaMemcpy(G.C.density, G.C.device, G.H.n_fields * G.H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);

  chprintf("\nSaving Snapshot: %d \n", nfile);

  // ensure the output-directory exists (try to create it if it doesn't exist)
  Ensure_Dir_Exists(write_manager.fname_template().effective_output_dir_path(nfile));

#ifdef RT
  // copy RT fields to CPU
  G.Rad.Copy_RT_Fields();
#endif

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
  G.Change_Cosmological_Frame_System(false);
#endif

  // this method call does most of the heavy lifting
  // -> recall that the writer manager was initialized at startup to manage a variable
  //    number of output types (e.g. field-dumps, particle-dumps, slices, etc.)
  // -> in this method call, the writer manager writes zero to all of the registered
  //    output types based on the value of nfile.
  write_manager.Apply_Writers(G, P, nfile);

#ifdef COSMOLOGY
  if (G.H.OUTPUT_SCALE_FACTOR || G.H.Output_Initial) {
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
  G.Change_Cosmological_Frame_System(true);
  chprintf("\n");
  G.H.Output_Now = false;
#endif

#ifdef HDF5
  // Cleanup HDF5
  H5close();
#endif

#ifdef MPI_CHOLLA
  MPI_Barrier(world);
#endif
}

void Grid3D::Write_Header(AttrRecorderInterface &attr_recorder) const
{
  // Single attributes first
  attr_recorder.record("gamma", gama);

  attr_recorder.record("Git Commit Hash", GIT_HASH);
  attr_recorder.record("Macro Flags", MACRO_FLAGS);
  attr_recorder.record("cholla", "");  // <- helps yt identify cholla outputs

  // Numeric Attributes
  attr_recorder.record("t", H.t);
  attr_recorder.record("dt", H.dt);
  attr_recorder.record("n_step", H.n_step);
  attr_recorder.record("n_fields", H.n_fields);
  attr_recorder.record("time_unit", double{TIME_UNIT});
  attr_recorder.record("length_unit", double{LENGTH_UNIT});
  attr_recorder.record("mass_unit", double{MASS_UNIT});
  attr_recorder.record("velocity_unit", double{VELOCITY_UNIT});
  attr_recorder.record("density_unit", double{DENSITY_UNIT});
  attr_recorder.record("energy_unit", double{ENERGY_UNIT});

#ifdef MHD
  attr_recorder.record("magnetic_field_unit", double{MAGNETIC_FIELD_UNIT});
#endif  // MHD

#ifdef COSMOLOGY
  attr_recorder.record("H0", Cosmo.H0);
  attr_recorder.record("Omega_M", Cosmo.Omega_M);
  attr_recorder.record("Omega_L", Cosmo.Omega_L);
  attr_recorder.record("Current_z", Cosmo.current_z);
  attr_recorder.record("Current_a", Cosmo.current_a);
#endif

  // Now, do 3-element attributes

  // todo: we should stop narrowing the datatype from ptrdiff_t to int
  int dims[3];
#ifndef MPI_CHOLLA
  dims[0] = H.nx_real;
  dims[1] = H.ny_real;
  dims[2] = H.nz_real;
#else
  dims[0] = nx_global;
  dims[1] = ny_global;
  dims[2] = nz_global;
#endif

  attr_recorder.record_arr("dims", dims, 3);

#ifdef MPI_CHOLLA
  attr_recorder.record_triple("dims_local", H.nx_real, H.ny_real, H.nz_real);

  // todo: we should stop narrowing the datatype from ptrdiff_t to int
  int offset[3];
  offset[0] = nx_local_start;
  offset[1] = ny_local_start;
  offset[2] = nz_local_start;

  attr_recorder.record_arr("offset", offset, 3);

  attr_recorder.record_triple("nprocs", nproc_x, nproc_y, nproc_z);
#endif

  attr_recorder.record_triple("bounds", H.xbound, H.ybound, H.zbound);
  attr_recorder.record_triple("domain", H.xdglobal, H.ydglobal, H.zdglobal);
  attr_recorder.record_triple("dx", H.dx, H.dy, H.dz);
}

#ifdef HDF5

H5Space1D &H5Space1D::operator=(H5Space1D &&other) noexcept
{
  std::swap(this->dim_, other.dim_);
  std::swap(this->id_, other.id_);
  return *this;
}

H5Space1D &H5Space1D::ensure_dim(hsize_t dim)
{
  if (this->id_ == H5I_INVALID_HID) {  // <- first time setting dim
    this->dim_ = std::make_unique<hsize_t>(dim);
  } else if (*this->dim_ == dim) {  // <- dim isn't changing
    return *this;
  } else {
    H5Sclose(this->id_);
    *this->dim_ = dim;
  }
  this->id_ = H5Screate_simple(1, this->dim_.get(), nullptr);
  CHOLLA_ASSERT(this->id_ != H5I_INVALID_HID, "dataspace init error");
  return *this;
}

hid_t H5AttrRecorder::make_attr_1d_(const char *name, hid_t type_id, hsize_t n_elem)
{
  hid_t dataspace_id = this->cached_dataspace_.ensure_dim(n_elem).id();
  CHOLLA_ASSERT(name != nullptr, "name must not be a nullptr");
  hid_t attribute_id = H5Acreate(this->file_id_, name, type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  CHOLLA_ASSERT(attribute_id != H5I_INVALID_HID, "error creating attribute");
  return attribute_id;
}

void H5AttrRecorder::record(const char *name, const char *value)
{
  CHOLLA_ASSERT(value != nullptr, "value can't be a nullptr");
  hid_t type_id      = this->stringType_;
  hid_t attribute_id = this->make_attr_1d_(name, type_id, 1);
  if (H5Awrite(attribute_id, type_id, &value) < 0) {
    CHOLLA_ERROR("error writing \"%s\" attribute", name);
  }
  if (H5Aclose(attribute_id) < 0) {
    CHOLLA_ERROR("error closing the \"%s\" attribute", name);
  }
}

void H5AttrRecorder::record_arr(const char *name, const double *arr, int length)
{
  hid_t type_id      = H5T_NATIVE_DOUBLE;
  hid_t dest_type_id = H5T_IEEE_F64LE;
  hid_t attribute_id = this->make_attr_1d_(name, dest_type_id, length);
  if (H5Awrite(attribute_id, type_id, arr) < 0) {
    CHOLLA_ERROR("error writing \"%s\" attribute", name);
  }
  if (H5Aclose(attribute_id) < 0) {
    CHOLLA_ERROR("error closing the \"%s\" attribute", name);
  }
}

void H5AttrRecorder::record_arr(const char *name, const int *arr, int length)
{
  hid_t type_id = H5T_NATIVE_INT;
  // the following was picked for historical consistency
  // -> in reality, we probably want to make sure the output size is at least
  //    as big as the native int (yes, it's usually 32-bit, but not guaranteed)
  // -> is there a compelling reason why this is big endian?
  hid_t dest_type_id = H5T_STD_I32LE;
  hid_t attribute_id = this->make_attr_1d_(name, dest_type_id, length);
  if (H5Awrite(attribute_id, type_id, arr) < 0) {
    CHOLLA_ERROR("error writing \"%s\" attribute", name);
  }
  if (H5Aclose(attribute_id) < 0) {
    CHOLLA_ERROR("error closing the \"%s\" attribute", name);
  }
}

void H5AttrRecorder::record_arr(const char *name, const long *arr, int length)
{
  hid_t type_id      = H5T_NATIVE_INT;
  hid_t dest_type_id = H5T_STD_I64LE;
  hid_t attribute_id = this->make_attr_1d_(name, dest_type_id, length);
  if (H5Awrite(attribute_id, type_id, arr) < 0) {
    CHOLLA_ERROR("error writing \"%s\" attribute", name);
  }
  if (H5Aclose(attribute_id) < 0) {
    CHOLLA_ERROR("error closing the \"%s\" attribute", name);
  }
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
  char sbuffer[1024];

#if defined HDF5
  filename += ".h5";
#endif  // HDF5
// for now assumes you will run on the same number of processors
#ifdef MPI_CHOLLA
  #ifdef TILED_INITIAL_CONDITIONS
  sprintf(sbuffer, "%sics_%dMpc_%d.h5", P.indir, (int)P.tile_length / 1000,
          H.nx_real);  // Everyone reads the same file
  filename = sbuffer;
  #else   // TILED_INITIAL_CONDITIONS is not defined
  filename += "." + std::to_string(procID);
  #endif  // TILED_INITIAL_CONDITIONS
#endif    // MPI_CHOLLA

#if defined HDF5
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
#endif  // HDF5
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

  #if defined(PRINT_INITIAL_STATS) && defined(COSMOLOGY)
/*! \fn void Print_Stats(Grid3D &G)
 *  \brief Compute stats for a grid. */
void Print_Stats(Grid3D &G)
{
  // Synchronize
  cudaMemcpy(G.C.density, G.C.device, G.H.n_fields * G.H.n_cells * sizeof(Real), cudaMemcpyDeviceToHost);
  // Write data
  G.Print_Grid_Stats();
}

/*! \fn void Print_Grid_Stats(void)
 *  \brief Compute stats for grid properties. */
void Grid3D::Print_Grid_Stats(void)
{
  int i, j, k, id, buf_id;
  Real mean_l, min_l, max_l;
  Real mean_g, min_g, max_g;

  // Print several interesting numbers

  // Density stats
  mean_l = 0;
  min_l  = 1e65;
  max_l  = -1;
  // Do density first
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += C.density[id];
        max_l = std::max(max_l, C.density[id]);
        min_l = std::min(min_l, C.density[id]);
      }
    }
  }
  mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real));

    #ifdef MPI_CHOLLA
  mean_g = ReduceRealAvg(mean_l);
  max_g  = ReduceRealMax(max_l);
  min_g  = ReduceRealMin(min_l);
  mean_l = mean_g;
  max_l  = max_g;
  min_l  = min_g;
    #endif  // MPI_CHOLLA
  chprintf("Density  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3] \n", mean_l, min_l, max_l);

  // Momentum stats

  // x momenta
  mean_l = 0;
  min_l  = 1e65;
  max_l  = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += std::abs(C.momentum_x[id]);
        max_l = std::max(max_l, std::abs(C.momentum_x[id]));
        min_l = std::min(min_l, std::abs(C.momentum_x[id]));
      }
    }
  }
  mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real));

    #ifdef MPI_CHOLLA
  mean_g = ReduceRealAvg(mean_l);
  max_g  = ReduceRealMax(max_l);
  min_g  = ReduceRealMin(min_l);
  mean_l = mean_g;
  max_l  = max_g;
  min_l  = min_g;
    #endif  // MPI_CHOLLA
  chprintf(" abs(Momentum X)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l);

  // y momenta
  mean_l = 0;
  min_l  = 1e65;
  max_l  = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += std::abs(C.momentum_y[id]);
        max_l = std::max(max_l, std::abs(C.momentum_y[id]));
        min_l = std::min(min_l, std::abs(C.momentum_y[id]));
      }
    }
  }
  mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real));

    #ifdef MPI_CHOLLA
  mean_g = ReduceRealAvg(mean_l);
  max_g  = ReduceRealMax(max_l);
  min_g  = ReduceRealMin(min_l);
  mean_l = mean_g;
  max_l  = max_g;
  min_l  = min_g;
    #endif  // MPI_CHOLLA
  chprintf(" abs(Momentum Y)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l);

  // z momenta
  mean_l = 0;
  min_l  = 1e65;
  max_l  = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += std::abs(C.momentum_z[id]);
        max_l = std::max(max_l, std::abs(C.momentum_z[id]));
        min_l = std::min(min_l, std::abs(C.momentum_z[id]));
      }
    }
  }
  mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real));

    #ifdef MPI_CHOLLA
  mean_g = ReduceRealAvg(mean_l);
  max_g  = ReduceRealMax(max_l);
  min_g  = ReduceRealMin(min_l);
  mean_l = mean_g;
  max_l  = max_g;
  min_l  = min_g;
    #endif  // MPI_CHOLLA
  chprintf(" abs(Momentum Z)  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km s^-1] \n", mean_l, min_l, max_l);

  // Energy
  mean_l = 0;
  min_l  = 1e65;
  max_l  = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += C.Energy[id];
        max_l = std::max(max_l, C.Energy[id]);
        min_l = std::min(min_l, C.Energy[id]);
      }
    }
  }
  mean_l /= ((H.nz_real) * (H.ny_real) * (H.nx_real));

    #ifdef MPI_CHOLLA
  mean_g = ReduceRealAvg(mean_l);
  max_g  = ReduceRealMax(max_l);
  min_g  = ReduceRealMin(min_l);
  mean_l = mean_g;
  max_l  = max_g;
  min_l  = min_g;
    #endif  // MPI_CHOLLA
  chprintf(" Energy  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ]\n", mean_l, min_l, max_l);

  Real temp, temp_max_l, temp_min_l, temp_mean_l;
  Real temp_min_g, temp_max_g, temp_mean_g;
  Real gase, vx, vy, vz;
  temp_mean_l = 0;
  temp_min_l  = 1e65;
  temp_max_l  = -1;
  mean_l      = 0;
  min_l       = 1e65;
  max_l       = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        vx     = C.momentum_x[id] / C.density[id];
        vy     = C.momentum_y[id] / C.density[id];
        vz     = C.momentum_z[id] / C.density[id];
        gase   = C.Energy[id] - 0.5 * C.density[id] * (vx * vx + vy * vy + vz * vz);
        mean_l += gase;
        max_l = std::max(max_l, gase);
        min_l = std::min(min_l, gase);

        temp = gase / C.density[id] * (gama - 1) * MP / KB * 1e10;
        temp_mean_l += temp;
        temp_max_l = std::max(temp_max_l, temp);
        temp_min_l = std::min(temp_min_l, temp);
      }
    }
  }
  mean_l /= (H.nz_real * H.ny_real * H.nx_real);
  temp_mean_l /= (H.nz_real * H.ny_real * H.nx_real);

    #ifdef MPI_CHOLLA
  mean_g      = ReduceRealAvg(mean_l);
  max_g       = ReduceRealMax(max_l);
  min_g       = ReduceRealMin(min_l);
  mean_l      = mean_g;
  max_l       = max_g;
  min_l       = min_g;
  temp_mean_g = ReduceRealAvg(temp_mean_l);
  temp_max_g  = ReduceRealMax(temp_max_l);
  temp_min_g  = ReduceRealMin(temp_min_l);
  temp_mean_l = temp_mean_g;
  temp_max_l  = temp_max_g;
  temp_min_l  = temp_min_g;
    #endif  // MPI_CHOLLA

  chprintf(" GasEnergyCalc  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ] \n", mean_l, min_l, max_l);
  chprintf(" TemperatureCalc  Mean: %f   Min: %f   Max: %f      [ K ] \n", temp_mean_l, temp_min_l, temp_max_l);

    #ifdef DE
  temp_mean_l = 0;
  temp_min_l  = 1e65;
  temp_max_l  = -1;
  mean_l      = 0;
  min_l       = 1e65;
  max_l       = -1;
  for (k = 0; k < H.nz_real; k++) {
    for (j = 0; j < H.ny_real; j++) {
      for (i = 0; i < H.nx_real; i++) {
        id     = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        buf_id = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        mean_l += C.GasEnergy[id];
        max_l = std::max(max_l, C.GasEnergy[id]);
        min_l = std::min(min_l, C.GasEnergy[id]);

        temp = C.GasEnergy[id] / C.density[id] * (gama - 1) * MP / KB * 1e10;
        temp_mean_l += temp;
        temp_max_l = std::max(temp_max_l, temp);
        temp_min_l = std::min(temp_min_l, temp);
      }
    }
  }
  mean_l /= (H.nz_real * H.ny_real * H.nx_real);
  temp_mean_l /= (H.nz_real * H.ny_real * H.nx_real);

      #ifdef MPI_CHOLLA
  mean_g      = ReduceRealAvg(mean_l);
  max_g       = ReduceRealMax(max_l);
  min_g       = ReduceRealMin(min_l);
  mean_l      = mean_g;
  max_l       = max_g;
  min_l       = min_g;
  temp_mean_g = ReduceRealAvg(temp_mean_l);
  temp_max_g  = ReduceRealMax(temp_max_l);
  temp_min_g  = ReduceRealMin(temp_min_l);
  temp_mean_l = temp_mean_g;
  temp_max_l  = temp_max_g;
  temp_min_l  = temp_min_g;
      #endif  // MPI_CHOLLA

  chprintf(" GasEnergyDE  Mean: %f   Min: %f   Max: %f      [ h^2 Msun kpc^-3 km^2 s^-2 ] \n", mean_l, min_l, max_l);
  chprintf(" TemperatureDE  Mean: %f   Min: %f   Max: %f      [ K ] \n", temp_mean_l, temp_min_l, temp_max_l);
    #endif  // DE
}
  #endif  // PRINT_INITIAL_STATS and COSMOLOGY

// this should work whether Cholla is configured with COOLING_GRACKLE or CHEMISTRY_GPU
// - earlier versions of this logic wouldn't work with Grackle
static void cosmo_init_chemical_species_(const Header &H, const FieldInfo &field_info, Real *host_field_ptr)
{
  auto get_ptr_or_abort = [&](const char *name) -> Real * {
    std::optional<int> maybe_id = field_info.field_id(name).value();
    if (maybe_id.has_value()) {
      return &host_field_ptr[H.n_cells * maybe_id.value()];
    } else {
      CHOLLA_ERROR("%s is not the name of a defined field", name);
    }
  };

  Real *density       = get_ptr_or_abort("density");
  Real *HI_density    = get_ptr_or_abort("HI_density");
  Real *HII_density   = get_ptr_or_abort("HII_density");
  Real *HeI_density   = get_ptr_or_abort("HeI_density");
  Real *HeII_density  = get_ptr_or_abort("HeII_density");
  Real *HeIII_density = get_ptr_or_abort("HeIII_density");
  Real *e_density     = get_ptr_or_abort("e_density");

  for (int k = 0; k < H.nz_real; k++) {
    for (int j = 0; j < H.ny_real; j++) {
      for (int i = 0; i < H.nx_real; i++) {
        int id            = (i + H.n_ghost) + (j + H.n_ghost) * H.nx + (k + H.n_ghost) * H.nx * H.ny;
        int buf_id        = k + j * (H.nz_real) + i * (H.nz_real) * (H.ny_real);
        HI_density[id]    = INITIAL_FRACTION_HI * density[id];
        HII_density[id]   = INITIAL_FRACTION_HII * density[id];
        HeI_density[id]   = INITIAL_FRACTION_HEI * density[id];
        HeII_density[id]  = INITIAL_FRACTION_HEII * density[id];
        HeIII_density[id] = INITIAL_FRACTION_HEIII * density[id];
        e_density[id]     = INITIAL_FRACTION_ELECTRON * density[id];
      }
    }
  }
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

  // load all of the hydro fields (include GasEnergy if using dual-energy formalism)
  for (int field_id : field_info.get_id_range(field::Kind::HYDRO)) {
    Real *dest_ptr                 = &C.host[field_id * H.n_cells];
    std::optional<std::string> tmp = field_info.field_name(field_id);
    if (!tmp.has_value()) {
      CHOLLA_ERROR("this should be unreachable");
    }
    std::string dset_name = "/" + tmp.value();
    Read_Grid_HDF5_Field(file_id, dataset_buffer, H, dest_ptr, dset_name.c_str());
  }

  // initialize a set of scalar fields that we want to skip
  std::unordered_set<std::string> skip_loading_scalar;
  #if (defined(COOLING_GRACKLE) || defined(CHEMISTRY_GPU)) && defined(COSMOLOGY)
  if (P.nfile == 0) {
    // overwrite skip_loading_scalar
    // -> we skip all primordial species fields, they're initialized in cosmo_init_chemical_species_
    // -> we also skip metal_density (presumably the field is set to 0 elsewhere?)
    skip_loading_scalar = {"HI_density",    "HII_density", "HeI_density",  "HeII_density",
                           "HeIII_density", "e_density",   "metal_density"};
    cosmo_init_chemical_species_(H, field_info, C.host);
  }
  #endif

  // try to load all of the scalars (that aren't within skip_loading_scalar)
  for (int field_id : field_info.get_id_range(field::Kind::PASSIVE_SCALAR)) {
    Real *dest_ptr                 = &C.host[field_id * H.n_cells];
    std::optional<std::string> tmp = field_info.field_name(field_id);
    if (!tmp.has_value()) {
      CHOLLA_ERROR("this should be unreachable");
    }
    std::string field_name = tmp.value();
    if (skip_loading_scalar.find(field_name) != skip_loading_scalar.end()) {
      continue;
    }
    std::string dset_name = "/" + field_name;
    Read_Grid_HDF5_Field(file_id, dataset_buffer, H, dest_ptr, dset_name.c_str());
  }

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

    #ifdef MPI_CHOLLA
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

    #ifdef MPI_CHOLLA
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

    #ifdef MPI_CHOLLA
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

void Ensure_Dir_Exists(std::string dir_path)
{
  if (Is_Root_Proc()) {
    // if the last character of outdir is not a '/', then the substring of
    // characters after the final '/' (or entire string if there isn't any '/')
    // is treated as a file-prefix
    //
    // this is accomplished here:
    std::filesystem::path path = std::filesystem::path(dir_path);

    if (!dir_path.empty()) {
      // try to create all directories specified within outdir (does nothing if
      // the directories already exist)
      std::error_code err_code;
      std::filesystem::create_directories(path, err_code);

      // confirm that an error-code wasn't set & that the path actually refers
      // to a directory (it's unclear from docs whether err-code is set in that
      // case)
      if (err_code or not std::filesystem::is_directory(path)) {
        CHOLLA_ERROR(
            "something went wrong while trying to create the path to the "
            "directory: %s",
            dir_path.c_str());
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
