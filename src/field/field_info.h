/*! \file
 *  Define machinery for accessing field information
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "../utils/FrozenKeyIdxBiMap.h"
#include "field_id.h"

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
  const std::vector<FieldId> id_vec_;

 public:
  explicit IdRange(const std::vector<FieldId>& id_vec) : id_vec_(id_vec) {}

  // the fact that the iterator aliases a const iterator of a std::vector is an
  // implementation detail
  using iterator = std::vector<FieldId>::const_iterator;

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
  // todo(before submitting PR): delete me!
  const utils::FrozenKeyIdxBiMap& get_field_id_map() const { return name_id_bimap_; }

  /*! \brief try to fetch the field_id associated with the @p field_name */
  std::optional<FieldId> field_id(std::string_view field_name) const
  {
    std::optional<int> tmp = name_id_bimap_.find(field_name);
    if (tmp.has_value()) {
      uint8_t pack_id = 0;  // todo: fix me when we support multiple packs
      return {FieldId(pack_id, static_cast<uint8_t>(*tmp))};
    }
    return std::nullopt;
  }

  /*! \brief try to look up slot-index of a field within a field-pack
   *
   *  \note This is a convenient way to check whether a field is associated with a pack
   */
  std::optional<int> slot_idx(uint8_t pack_id, std::string_view field_name) const
  {
    std::optional<FieldId> tmp = field_id(field_name);
    if (tmp.has_value() && tmp->pack_id == pack_id) {
      return {static_cast<int>(tmp->slot_idx)};
    }
    return std::nullopt;
  }
  std::optional<int> slot_idx(std::string_view pack_name, std::string_view field_name) const
  {
    uint8_t my_pack_id = pack_id(pack_name).value_or(static_cast<uint8_t>(n_packs()));
    return slot_idx(my_pack_id, field_name);
  }

  /*! \brief try to look up the field name from the field id */
  std::optional<std::string> field_name(FieldId field_id) const
  {
    if (field_id.slot_idx >= n_fields(field_id.pack_id)) {
      return std::nullopt;
    }
    // todo: fix me when we support multiple packs
    return std::optional<std::string>{name_id_bimap_.inverse_find(field_id.slot_idx)};
  }

  /*! \brief try to look up the associated pack name */
  std::optional<std::string> pack_name(FieldId field_id) const { return pack_name(field_id.pack_id); }
  std::optional<std::string> pack_name(uint8_t pack_id) const
  {
    // todo: fix me when we support multiple packs
    return (pack_id == 0) ? std::optional<std::string>{"fluid"} : std::nullopt;
  }

  /*! \brief try to lookup the associated pack_id */
  std::optional<uint8_t> pack_id(FieldId field_id) const
  {
    // external code is explicitly prohibitted from accessing internals of field_id
    // (gives us the freedom to change how field_id is implemented in the future)
    return field_id.pack_id;
  }
  std::optional<uint8_t> pack_id(std::string_view pack_name) const
  {
    // todo: fix me when we support multiple packs
    return (pack_name == "fluid") ? std::optional<uint8_t>{0} : std::nullopt;
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
  std::optional<bool> is_cell_centered(FieldId field_id) const
  {
    if (field_id.pack_id != 0 || field_id.slot_idx >= n_fields()) {
      return std::nullopt;
    }
    for (int id : magnetic_field_ids_) {
      if (field_id.slot_idx == id) {
        return std::optional<bool>{false};
      }
    }
    return std::optional<bool>{true};
  }

  /*! try to look up the IOBuf value associated with a field
   *
   *  \note We may want to revisit whether this actually should be tracked by FieldInfo in the future.
   */
  std::optional<field::IOBuf> io_buf(FieldId field_id) const
  {
    bool bad_id = (field_id.pack_id != 0 || field_id.slot_idx >= n_fields());
    return bad_id ? std::nullopt : std::optional<field::IOBuf>{io_buf_[field_id.slot_idx]};
  }

  /*! Returns the number of fields */
  int n_fields() const { return static_cast<int>(name_id_bimap_.size()); }

  /*! Returns the number of fields of a given category */
  int n_fields(field::Kind kind) const { return static_cast<int>(get_kind_ids_(kind).size()); }

  /*! \brief Returns the number of fields associated with a the specified pack_id */
  int n_fields(uint8_t pack_id) const { return (pack_id == 0) ? n_fields() : 0; }

  /*! \brief Returns the number of field-packs */
  int n_packs() const { return (n_fields() > 0) ? 1 : 0; }

  /*! Returns the first field_id corresponding to a passive scalar (if there are any) */
  std::optional<int> scalar_start() const
  {
    return scalar_field_ids_.empty() ? std::nullopt : std::optional<int>{scalar_field_ids_[0]};
  }

  /*! This returns a "range" over all ids
   *
   *  This might be used in a case like the following:
   *  \code{c++}
   *  for (FieldId field_id: field_info.get_id_range(field::Kind::HYDRO)) {
   *    // ...
   *  }
   *  \endcode
   */
  field::IdRange get_id_range(field::Kind kind) const
  {
    const std::vector<int>& tmp = get_kind_ids_(kind);
    std::vector<FieldId> v;
    v.reserve(tmp.size());
    for (int slot_idx : tmp) {
      FieldId f_id(0, static_cast<uint8_t>(slot_idx));
      v.emplace_back(std::move(f_id));
    }
    return field::IdRange(std::move(v));
  }
};