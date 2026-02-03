/*!
 * \file
 * Declares the FieldWriter type
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "../global/global.h"
#include "../grid/field_info.h"
#include "../grid/grid3D.h"
#include "../io/FnameTemplate.h"     // define FnameTemplate
#include "../io/ParameterMap.h"      // define ParameterMap
#include "../utils/basic_structs.h"  // VectorXYZ

namespace io
{

enum struct WriteCond { ALWAYS, REQUIRE_COMPLETE_DATA };

struct DatasetSpecEntry {
  int field_id;
  /// the dataset name. By convention, this is prefixed with a "/"
  std::string name;
  /// indicates whether we record values from the host or device buffers
  field::IOBuf io_buf;
  /// the condition for writing this dataset
  WriteCond condition;
};

struct DatasetSpec {
  /// describes properties about dataset creation for ordinary cell-centered fields
  std::vector<DatasetSpecEntry> cc_dataset_entries;
  /// indicates whether we should write magnetic fields
  ///
  /// \note Ideally, we would handle these a little more uniformly with other fields,
  /// but that's a task for another time
  std::optional<WriteCond> mhd_condition;
};

/*! \brief A callable that writes general grid data
 *
 *  \todo Maybe work to consolidate this with F32FieldWriter
 */
class FieldWriter
{
  DatasetSpec h5_dataset_spec_;

 public:
  FieldWriter() = delete;
  FieldWriter(ParameterMap &pmap, const FieldInfo &field_info);

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const;
};

/*! \brief A callable for writing 32-bit outputs of general grid data
 *
 *  \todo Maybe work to consolidate this with FieldWriter
 */
class F32FieldWriter
{
  hydro_utilities::VectorXYZ<bool> write_mag = {false, false, false};
  std::vector<DatasetSpecEntry> cc_dataset_entries;

 public:
  F32FieldWriter() = delete;
  F32FieldWriter(ParameterMap &pmap, const FieldInfo &field_info);

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const;
};

}  // namespace io