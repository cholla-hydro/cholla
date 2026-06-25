/*! \file
 *  \brief Declares logic pertaining to reading and writing data.
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <type_traits>  // std::is_same
#include <utility>      // std::swap, std::move

#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../io/AttrRecorderInterface.h"
#include "../io/FieldWriter.h"
#include "../io/FnameTemplate.h"
#include "../io/WriterManager.h"
#include "../utils/error_handling.h"

/*! Local function that designates whether we are using a root-process. It gives
 *  a sensible result regardless of whether we are using MPI
 */
inline bool Is_Root_Proc() { return procID == root; }

/* Compute stats for a grid. */
void Print_Stats(Grid3D& G);

/*! Write all data files
 *
 *  \param G the grid object
 *  \param P the parameter struct
 *  \param nfile the index corresponding to the current output
 *  \param writer_manager Manages the data writers.
 */
void Write_Data(Grid3D& G, struct Parameters P, int nfile, const io::WriterManager& write_manager);

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

/* Checks whether the directories referred to within outdir exist. Creates them
 * if they don't. It gracefully handles cases where outdir contains a prefix
 * for the output files.
 */
void Ensure_Dir_Exists(std::string dir_path);

#ifdef HDF5
// From io/io.cpp

/*! Encapsulates a simple 1D H5 dataspace
 *
 *  After a LOT of debugging, it turns out that we need to preserve the pointer used
 *  used to call @ref H5Screate_simple. This class does that for us in a convenient way
 *  for implementing @ref H5AttrRecorder
 */
class H5Space1D
{
  std::unique_ptr<hsize_t> dim_;
  hid_t id_;

 public:
  H5Space1D() : dim_(nullptr), id_{H5I_INVALID_HID} {}
  H5Space1D(hsize_t dim) : H5Space1D() { this->ensure_dim(dim); }
  H5Space1D(H5Space1D&& other) noexcept : H5Space1D() { *this = std::move(other); }

  /*! Move Assignment */
  H5Space1D& operator=(H5Space1D&& other) noexcept;

  ~H5Space1D()
  {
    if (this->id_ != H5I_INVALID_HID) H5Sclose(this->id_);
  }

  /*! get the dataspace id */
  hid_t id() const { return this->id_; }

  /*! ensure that the underlying dataspace dimension is `dim` (dataspace id may change) */
  H5Space1D& ensure_dim(hsize_t dim);
};

/*! Provides a nice wrapper around an hdf5 file handle for the purpose of recording
 *  attributes
 */
class H5AttrRecorder : public AttrRecorderInterface
{
  hid_t file_id_;
  hid_t stringType_;
  H5Space1D cached_dataspace_;

  hid_t make_attr_1d_(const char* name, hid_t type_id, hsize_t n_elem);

 public:
  H5AttrRecorder() = delete;

  explicit H5AttrRecorder(hid_t file_id)
  {
    this->file_id_    = file_id;
    this->stringType_ = H5Tcopy(H5T_C_S1);
    CHOLLA_ASSERT(H5Tset_size(this->stringType_, H5T_VARIABLE) >= 0, "error creating the string type");
  }

  ~H5AttrRecorder() override { H5Tclose(this->stringType_); }

  void record_arr(const char* name, const double* arr, int length) override;
  void record_arr(const char* name, const int* arr, int length) override;
  void record_arr(const char* name, const long* arr, int length) override;
  void record(const char* name, const char* val) override;

  // for historical consistency scalar arithmetic values are saved as 1-element arrays
  void record(const char* name, double val) override { this->record_arr(name, &val, 1); }
  void record(const char* name, int val) override { this->record_arr(name, &val, 1); }
  void record(const char* name, long val) override { this->record_arr(name, &val, 1); }
};

herr_t Read_HDF5_Dataset(hid_t file_id, double* dataset_buffer, const char* name);
herr_t Read_HDF5_Dataset(hid_t file_id, float* dataset_buffer, const char* name);

herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, double* dataset_buffer, const char* name);
herr_t Write_HDF5_Dataset(hid_t file_id, hid_t dataspace_id, float* dataset_buffer, const char* name);

/* \brief After HDF5 reads data into a buffer, remap and write to grid buffer. */
void Fill_Grid_From_HDF5_Buffer(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                Real* hdf5_buffer, Real* grid_buffer);

/*! Data moves from host grid_buffer to dataset_buffer to hdf5 file */
void Write_Grid_HDF5_Field_CPU(Header H, hid_t file_id, Real* dataset_buffer, Real* grid_buffer, const char* name);

/*! Data moves from device_grid_buffer to device_hdf5_buffer to dataset_buffer to hdf5 file */
void Write_Grid_HDF5_Field_GPU(Header H, hid_t file_id, Real* dataset_buffer, Real* device_hdf5_buffer,
                               Real* device_grid_buffer, const char* name);

/*! Generic field writer from GPU */
void Write_Generic_HDF5_Field_GPU(int nx, int ny, int nz, int nx_real, int ny_real, int nz_real, int n_ghost,
                                  hid_t file_id, Real *dataset_buffer, Real *device_hdf5_buffer, Real *source_buffer,
                                  const char *name);

// From io/io_gpu.cu
// Use GPU to pack source -> device_buffer, then copy device_buffer -> buffer,
// then write HDF5 field
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         float* buffer, float* device_buffer, Real* source, const char* name, int mhd_direction = -1);
void Write_HDF5_Field_3D(int nx, int ny, int nx_real, int ny_real, int nz_real, int n_ghost, hid_t file_id,
                         double* buffer, double* device_buffer, Real* source, const char* name, int mhd_direction = -1);
#endif
