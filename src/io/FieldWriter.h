/*!
 * \file
 * Declares the FieldWriter type
 */

#pragma once

#include <string>
#include <vector>

#include "../global/global.h"
#include "../grid/field_info.h"
#include "../grid/grid3D.h"
#include "../io/FnameTemplate.h"  // define FnameTemplate
#include "../io/ParameterMap.h"   // define ParameterMap

namespace io
{

enum struct WriteCond { ALWAYS, REQUIRE_COMPLETE_DATA };

struct DatasetSpec {
  int field_id;
  /// the dataset name. By convention, this is prefixed with a "/"
  std::string name;
  /// indicates whether we record values from the host or device buffers
  field::IOBuf io_buf;
  /// the condition for writing this dataset
  WriteCond condition;
};

/*! \brief A callable that writes general grid data
 */
class FieldWriter
{
  std::vector<DatasetSpec> h5_dataset_spec_;

 public:
  FieldWriter() = delete;
  FieldWriter(ParameterMap &pmap, const FieldInfo &field_info);

  /*! A callable method that writes a rotated projection of the grid data to file.
   */
  void operator()(Grid3D &G, Parameters P, int nfile, const FnameTemplate &fname_template) const;
};

}  // namespace io