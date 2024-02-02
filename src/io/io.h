#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>

#include "../global/global.h"
#include "../grid/grid3D.h"

/* Write the data */
void Write_Data(Grid3D& G, struct Parameters P, int nfile);

/* Output the grid data to file. */
void Output_Data(Grid3D& G, struct Parameters P, int nfile);

/* Output the grid data to file as 32-bit floats. */
void Output_Float32(Grid3D& G, struct Parameters P, int nfile);

/* Output a projection of the grid data to file. */
void Output_Projected_Data(Grid3D& G, struct Parameters P, int nfile);

/* Output a rotated projection of the grid data to file. */
void Output_Rotated_Projected_Data(Grid3D& G, struct Parameters P, int nfile);

/* Output xy, xz, and yz slices of the grid data to file. */
void Output_Slices(Grid3D& G, struct Parameters P, int nfile);

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

void Create_Log_File(struct Parameters P);

void Write_Message_To_Log_File(const char* message);

void Write_Debug(Real* Value, const char* fname, int nValues, int iProc);

/* Lightweight object designed to centralize the file-naming logic (& any associated configuration).
 *
 * Cholla pathnames traditionally followed the following template:
 *     "{outdir}{nfile}{pre_extension_suffix}{extension}.{proc_id}"
 * where each curly-braced token represents a different variable. In detail:
 *   - `{outdir}` is the parameter from the parameter file. The historical behavior (that we currently
 *     maintain), if this is non-empty, then all charaters following the last '/' are treated as a
 *     prefix to the output file name (if there aren't any '/' characters, then the whole string is
 *     effectively a prefix.
 *   - `{nfile}` is the current file-output count.
 *   - `{pre_extension_suffix}` is the pre-hdf5-extension suffix. It's the suffix that precedes the
 *     file extension (or `{extension}`)
 *   - `{extension}` is the filename extension. Examples include ".h5" or ".bin" or ".txt".
 *   - `{proc_id}` represents the process-id that held the data that will be written to this file.
 *     Previously, in non-MPI runs, this was omitted.
 *
 * Instances can be configured to support the following newer file-naming template
 *    "{outdir}/{nfile}/{nfile}{pre_extension_suffix}{extension}.{proc_id}"
 * where the the significance of each curly-braced token is largely unchanged. There are 2 things
 * worth noting:
 *   - all files written at a single simulation-cycle are now grouped in a single directory
 *   - `{outdir}` never specifies a file prefix. When `{outdir}` is empty, it is treated as "./".
 *     Otherwise, we effectively append '/' to the end of `{outdir}`
 *
 * \note
 * This could probably pull double-duty and get reused with infile.
 */
class FnameTemplate
{
 public:
  FnameTemplate() = delete;

  FnameTemplate(bool separate_cycle_dirs, std::string outdir)
      : separate_cycle_dirs_(separate_cycle_dirs), outdir_(std::move(outdir))
  {
  }

  FnameTemplate(const Parameters& P) : FnameTemplate(not P.legacy_flat_outdir, P.outdir) {}

  /* Specifies whether separate cycles are written to separate directories */
  bool separate_cycle_dirs() const noexcept { return separate_cycle_dirs_; }

  /* Returns the effective output-directory used for outputs at a given simulation-cycle */
  std::string effective_output_dir_path(int nfile) const noexcept;

  /* format the file path */
  std::string format_fname(int nfile, const std::string& pre_extension_suffix) const noexcept;

  std::string format_fname(int nfile, int file_proc_id, const std::string& pre_extension_suffix) const noexcept;

 private:
  bool separate_cycle_dirs_;
  std::string outdir_;
};

/* Checks whether the directories referred to within outdir exist. Creates them
 * if they don't. It gracefully handles cases where outdir contains a prefix
 * for the output files.
 */
void Ensure_Dir_Exists(std::string dir_path);

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
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         float* buffer, float* device_buffer, Real* source, const char* name, int mhd_direction = -1);
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         double* buffer, double* device_buffer, Real* source, const char* name, int mhd_direction = -1);
#endif
