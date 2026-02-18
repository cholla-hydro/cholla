/*! \file
 *  Define machinery for accessing field information
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../utils/FrozenKeyIdxBiMap.h"

namespace field
{

// note: HYDRO includes GasEnergy (if present)
enum class Kind { HYDRO, PASSIVE_SCALAR, MAGNETIC };

/*! Specifies which buffer to use for IO */
enum class IOBuf { HOST, DEVICE };

/*! Specifies centering */

/*! This is a "range" in the C++ 20 sense
 *
 *  See \ref FieldInfo::get_id_range for an example
 */
class IdRange
{
  const std::vector<int>& id_vec_;

 public:
  explicit IdRange(const std::vector<int>& id_vec) : id_vec_(id_vec) {}

  // the fact that the iterator aliases a const iterator of a std::vector is an
  // implementation detail
  using iterator = std::vector<int>::const_iterator;

  iterator begin() const { return id_vec_.begin(); }
  iterator end() const { return id_vec_.end(); }
};

}  // namespace field

/*! Dynamically describes the available fields and associated properties
 */
class FieldInfo
{
  utils::FrozenKeyIdxBiMap name_id_bimap_;
  std::vector<int> hydro_field_ids_;
  std::vector<int> scalar_field_ids_;
  std::vector<int> magnetic_field_ids_;
  std::vector<field::IOBuf> io_buf_;

  // We make the default-constructor private to force the use of the factory method
  FieldInfo() = default;

  /*! return a reference to the internal vector of field ids corresponding to
   *  @ref field::Kind
   */
  const std::vector<int>& get_kind_ids_(field::Kind kind) const;

 public:
  /*! Factory method
   *
   *  Ideally, we would make it possible to customize the active scalars, but that's a
   *  topic for the future.
   */
  static FieldInfo create();

  FieldInfo(FieldInfo&&)            = default;
  FieldInfo& operator=(FieldInfo&&) = default;

  // we delete copy constructor and copy-assignment to prevent accidental copies
  // (of course move constructors/move assignment remain possible)
  // In the unlikely event we decide to support copies, this can always change later...
  FieldInfo(const FieldInfo&)            = delete;
  FieldInfo& operator=(const FieldInfo&) = delete;

  /*! Get the underlying mapping object between field names and ids */
  const utils::FrozenKeyIdxBiMap& get_field_id_map() const { return name_id_bimap_; }

  /*! try to lookup the field_id associated with the field_name */
  std::optional<int> field_id(const char* field_name) const { return name_id_bimap_.find(field_name); }
  std::optional<int> field_id(std::string_view field_name) const { return name_id_bimap_.find(field_name); }

  /*! try to look up the field name from the field id */
  std::optional<std::string> field_name(int field_id) const
  {
    bool bad_id = (field_id < 0 || field_id >= n_fields());
    return bad_id ? std::nullopt : std::optional<std::string>{name_id_bimap_.inverse_find(field_id)};
  }

  /*! try to look up whether the field id refers to a cell-centered field
   *
   *  \note
   *  It may be more useful to return a value that directly specifies whether a field is
   *  cell-center, x-face-centered, y-face-centered, z-face-centered. It might be
   *  convenient to specify this with a @ref hydro_utilities::VectorXYZ<int>. For
   *  example, `{0,0,0}` could represent a cell-centered value and `{1,0,0}`, or maybe
   *  `{-1, 0, 0}` (we need to think about conventions), could denote a field centered
   *  on x-faces.
   */
  std::optional<bool> is_cell_centered(int field_id)
  {
    if (field_id < 0 || field_id >= n_fields()) {
      return std::nullopt;
    }
    for (int id : magnetic_field_ids_) {
      if (field_id == id) {
        return std::optional<bool>{false};
      }
    }
    return std::optional<bool>{true};
  }

  /*! try to look up the IOBuf value associated with a field
   *
   *  \note We may want to revisit whether this actually should be tracked by FieldInfo in the future.
   */
  std::optional<field::IOBuf> io_buf(int field_id) const
  {
    bool bad_id = (field_id < 0 || field_id >= n_fields());
    return bad_id ? std::nullopt : std::optional<field::IOBuf>{io_buf_[field_id]};
  }

  /*! Returns the number of fields */
  int n_fields() const { return static_cast<int>(name_id_bimap_.size()); }

  /*! Returns the number of fields of a given category */
  int n_fields(field::Kind kind) const { return static_cast<int>(get_kind_ids_(kind).size()); }

  /*! Returns the first field_id corresponding to a passive scalar (if there are any) */
  std::optional<int> scalar_start() const
  {
    return scalar_field_ids_.empty() ? std::nullopt : std::optional<int>{scalar_field_ids_[0]};
  }

  /*! This returns a "range" over all ids
   *
   *  This might be used in a case like the following:
   *  \code{c++}
   *  for (int field_id: field_info.get_id_range(field::Kind::HYDRO)) {
   *    // ...
   *  }
   *  \endcode
   */
  field::IdRange get_id_range(field::Kind kind) const { return field::IdRange(get_kind_ids_(kind)); }
};