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

/*! \brief A callable type that writes general grid data
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 */
class FieldWriter
{
  DatasetSpec h5_dataset_spec_;

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