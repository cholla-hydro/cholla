#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>

#include "../global/global.h"
#include "../grid/grid3D.h"

/* Write the data */
void WriteData(Grid3D& G, struct parameters P, int nfile);

/* Output the grid data to file. */
void OutputData(Grid3D& G, struct parameters P, int nfile);

/* Output the grid data to file as 32-bit floats. */
void OutputFloat32(Grid3D& G, struct parameters P, int nfile);

/* Output a projection of the grid data to file. */
void OutputProjectedData(Grid3D& G, struct parameters P, int nfile);

/* Output a rotated projection of the grid data to file. */
void OutputRotatedProjectedData(Grid3D& G, struct parameters P, int nfile);

/* Output xy, xz, and yz slices of the grid data to file. */
void OutputSlices(Grid3D& G, struct parameters P, int nfile);

/* MPI-safe printf routine */
int chprintf(const char* __restrict sdata, ...);

/*!
 * \brief Convert a floating point number to a string such that it can be
 * exactly deserialized back from a string to the same floating point number.
 *
 * \tparam T Any floating point type
 * \param[in] input The floating point number to convert
 * \return std::string The string representation of the input floating point
 */
template <typename T>
std::string to_string_exact(T const& input)
{
  std::stringstream output;
  output << std::setprecision(std::numeric_limits<T>::max_digits10);
  output << input;
  return output.str();
}

void Create_Log_File(struct parameters P);

void Write_Message_To_Log_File(const char* message);

void write_debug(Real* Value, const char* fname, int nValues, int iProc);

/* Checks whether the directories referred to within outdir exist. Creates them
 * if they don't. It gracefully handles cases where outdir contains a prefix
 * for the output files.
 */
void Ensure_Outdir_Exists(std::string outdir);

#ifdef HDF5
// From io/io.cpp

herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, double* attribute, const char* name);
herr_t Write_HDF5_Attribute(hid_t file_id, hid_t dataspace_id, int* attribute, const char* name);

herr_t Read_HDF5_Dataset(hid_t file_id, double* dataset_buffer, const char* name);
herr_t Read_HDF5_Dataset(hid_t file_id, float* dataset_buffer, const char* name);

herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, double* dataset_buffer, const char* name);
herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, float* dataset_buffer, const char* name);

/* \brief After HDF5 reads data into a buffer, remap and write to grid buffer. */
void Fill_Grid_From_HDF5_Buffer(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                Real* hdf5_buffer, Real* grid_buffer);

// From io/io_gpu.cu
// Use GPU to pack source -> device_buffer, then copy device_buffer -> buffer,
// then write HDF5 field
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, float* buffer,
                      float* device_buffer, Real* source, const char* name, int mhd_direction = -1);
void WriteHDF5Field3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id, double* buffer,
                      double* device_buffer, Real* source, const char* name, int mhd_direction = -1);
#endif
