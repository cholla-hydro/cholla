/*!
 * \file
 * Declares the FieldWriter type
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../global/global.h"
#include "../grid/field_info.h"
#include "../grid/grid3D.h"
#include "../io/FnameTemplate.h"  // define FnameTemplate
#include "../io/LazyScratchBuf.h"
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
  hydro_utilities::VectorXYZ<bool> write_mag = {false, false, false};
};

/*! \brief A callable type that writes general grid data
 *
 *  For more context, a "callable" object is sometimes called a "functor." Essentially
 *  a "callable" object carries around state and can be called like a function.
 *
 *  \todo Maybe work to consolidate this with F32FieldWriter
 */
class FieldWriter
{
  DatasetSpec h5_dataset_spec_;
  /*! this is tracked in a pointer so we can mutate the buffer even in const methods
   *
   *  \note
   *  I'm not thrilled that this is a shared pointer, but that seems to be the only
   *  viable solution since std::function requires that this class is copy-constructible
   */
  std::shared_ptr<LazyScratchBuf> lazy_scratch_buf_;

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

/*! \brief A callable for writing 32-bit outputs of general grid data
 *
 *  \todo Maybe work to consolidate this with FieldWriter
 */
class F32FieldWriter
{
  DatasetSpec dataset_spec_;
  /*! this is tracked in a pointer so we can mutate the buffer even in const methods
   *
   *  \note
   *  I'm not thrilled that this is a shared pointer, but that seems to be the only
   *  viable solution since std::function requires that this class is copy-constructible
   */
  std::shared_ptr<LazyScratchBuf> lazy_scratch_buf_;

 public:
  F32FieldWriter() = delete;
  F32FieldWriter(ParameterMap &pmap, const FieldInfo &field_info);

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