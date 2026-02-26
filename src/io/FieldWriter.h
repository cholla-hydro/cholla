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

enum struct FileFormat { TEXT, H5_NATIVE_PRECISION };

/*! Specifies the condition for writing data to disk. */
enum struct WriteCond { ALWAYS, REQUIRE_COMPLETE_DATA };

/*! Specifies the information about a "dataset" that will be written to disk
 *
 *  At the time of writing, this is only useful for cell-centered fields
 */
struct DatasetSpecEntry {
  /// the id of the field that will be written
  int field_id;
  /// the name used to refer to the field data in the output file
  ///
  /// The precise interpretation depends upon context:
  /// - For HDF5 datasets, this is prefixed with a "/"
  /// - For text-file outputs, this is the name of the column that holds the data
  std::string name;
  /// indicates whether we record values from the host or device buffers
  field::IOBuf io_buf;
  /// the condition for writing this dataset
  WriteCond condition;

  // the following constructor is defined in order to make this type work with
  // std::vector::emplace_back. Delete it, once we require C++20 or newer
#if __cpp_aggregate_paren_init < 201902L
  DatasetSpecEntry(int field_id, const std::string &name, field::IOBuf io_buf, WriteCond condition)
      : field_id{field_id}, name{name}, io_buf{io_buf}, condition{condition}
  {
  }
#endif
};

/*! Temporary type for tracking field-writer configuration
 *
 *  \note
 *  In PR#469, we'll store the members of this struct directly within @ref FieldWriter
 */
struct DatasetSpec {
  /// describes properties about dataset creation for ordinary cell-centered fields
  std::vector<DatasetSpecEntry> cc_dataset_entries;
  /// indicates whether we should write magnetic fields
  ///
  /// \note Ideally, we would handle these a little more uniformly with other fields,
  /// but that's a task for another time
  std::optional<WriteCond> mhd_condition;
};

/*! \brief A callable type that writes general grid data
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 */
class FieldWriter
{
  FileFormat file_format_;
  DatasetSpec dataset_spec_;

 public:
  FieldWriter() = delete;
  FieldWriter(ParameterMap &pmap, const FieldInfo &field_info);

  /*! Writes the field data to disk.
   *
   *  \note
   *  In case you are unaware, this overloads the "function call operator". If we have an
   *  instance, `obj`, then you call this method by invoking `obj(G, P, nfile, fname_template)`.
   *  In python, this method would be called `__call__`
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const;
};

}  // namespace io
