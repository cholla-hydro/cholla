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
#include "iter.h"

// define constructs that code outside of the field submodule never directly encounters
namespace field_detail
{

/*! \brief specifies information about a single field pack */
struct PackInfo {
  /// slice `flat_idx` values for each field in the field pack
  ///
  /// @note
  /// `flat_idx` used for implementing @ref FieldInfo -- for more information, see the
  /// section in its docstring about implementation details.
  IdxSlc flat_idx_slc;
};

}  // namespace field_detail

namespace field
{

// note: HYDRO includes GasEnergy (if present)
enum class Kind { HYDRO, PASSIVE_SCALAR, MAGNETIC };

}  // namespace field

/*! \brief Queryable object describing available fields and associated properties
 *
 *  This allows querying of properties about fields or packs of fields.
 *
 *  Every field belongs to one field-pack. Fields are ordered within a field pack. The
 *  index of a given field in a field pack is called the ``slot_idx``.
 *
 *  You can have one or more host data register register and one or more device data
 *  registers for each field in a field pack (management of the memory allocations that
 *  store field data is handled outside of FieldInfo). For a given register of a field
 *  pack, the data of all fields in that pack is stored in a contiguous memory
 *  allocation. This allows solvers that know about the `slot_idx` of each field in a
 *  field-pack at compile-time (like the hydro/mhd solver) to just be passed a pointer
 *  to the full field pack (this facillitates certain optimizations).
 *
 *  Each field has a unique field_id. This is a handle encoded inside the opaque
 *  \ref FieldId type (opaque means that the internal representation of a \ref FieldId
 *  is obscured from non-field machinery -- allowing us to change in the future).
 *
 *  Each field-pack has a unique pack_id. External machinery *should* generally treat
 *  this as if its a handle type.
 *
 *  Implementation Details
 *  ======================
 *  Under the hood, a field currently maps to a `flat_idx` in addition to a unique
 *  \ref FieldId. Whereas external code may interact with \ref FieldId objects (i.e.
 *  they are handed existing objects, they can copy them and then pass them back to the
 *  field machinery), a `flat_idx` is only used inside of \ref FieldInfo (and perhaps to
 *  implement interators/ranges if we're feeling ambitious in the future).
 *
 *  Current definition of a `flat_idx`:
 *  - it corresponds to the index of a field in a flat, contiguous sequence of all field
 *    names. This list is constructed by concatenating the sequences of field names from
 *    each field pack.
 *  - the `flat_idx` of a given field is the sum of the field's `slot_idx` and the
 *    `flat_idx` corresponding to `slot_idx = 0` of the field's field pack.
 */
class FieldInfo
{
  /// bidirectional mapping between field names and the corresponding flat_idx
  utils::FrozenKeyIdxBiMap name_id_bimap_;

  /// bidirectional mapping between pack names and the corresponding pack_id
  utils::FrozenKeyIdxBiMap pname_id_bimap_;

  /// specifies the buffer to use for IO
  std::vector<MemSpace> io_buf_;

  /// tracks properties of each field pack
  ///
  /// @note always has the same number of entries as pname_id_bimap_
  std::vector<field_detail::PackInfo> pack_info_;

  // We make the default-constructor private to force the use of the factory method
  FieldInfo() = default;

  const std::optional<std::size_t> flat_idx_from_FieldId_(FieldId id) const
  {
    if (id.slot_idx >= n_fields(id.pack_id)) {
      return std::nullopt;
    }
    const field_detail::IdxSlc& slc = pack_info_[id.pack_id].flat_idx_slc;
    return {slc.start() + id.slot_idx};
  }

  const std::optional<FieldId> FieldId_from_flat_idx_(std::size_t flat_idx) const
  {
    // if number of packs is LARGE, linear search gets slow & we should refactor
    uint8_t n_packs = static_cast<uint8_t>(pack_info_.size());
    for (uint8_t pack_id = 0; pack_id < n_packs; pack_id++) {
      const field_detail::IdxSlc& slc = pack_info_[pack_id].flat_idx_slc;
      if (slc.stop() <= flat_idx) continue;
      FieldId out(pack_id, static_cast<uint8_t>(flat_idx - slc.start()));
      return {out};
    }
    return std::nullopt;
  }

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
      return FieldId_from_flat_idx_(tmp.value());
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
    std::optional<std::size_t> tmp = flat_idx_from_FieldId_(field_id);
    if (tmp.has_value()) {
      return std::optional<std::string>{name_id_bimap_.inverse_find(tmp.value())};
    }
    return std::nullopt;
  }

  /*! \brief try to look up the associated pack name */
  std::optional<std::string> pack_name(FieldId field_id) const { return pack_name(field_id.pack_id); }
  std::optional<std::string> pack_name(uint8_t pack_id) const
  {
    if (static_cast<int>(pack_id) < n_packs()) {
      return std::optional<std::string>{pname_id_bimap_.inverse_find(pack_id)};
    }
    return std::nullopt;
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
    std::optional<int> tmp = pname_id_bimap_.find(pack_name);
    if (tmp.has_value()) return std::optional<uint8_t>{tmp.value()};
    return std::nullopt;
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
    for (FieldId id : get_id_range(field::Kind::MAGNETIC)) {
      if (field_id == id) return std::optional<bool>{false};
    }
    if (flat_idx_from_FieldId_(field_id).has_value()) {
      return std::optional<bool>{true};
    } else {
      return std::nullopt;
    }
  }

  /*! try to look up the IOBuf value associated with a field
   *
   *  \note We may want to revisit whether this actually should be tracked by FieldInfo in the future.
   */
  std::optional<MemSpace> io_buf(FieldId field_id) const
  {
    bool bad_id = (field_id.pack_id != 0 || field_id.slot_idx >= n_fields());
    return bad_id ? std::nullopt : std::optional<MemSpace>{io_buf_[field_id.slot_idx]};
  }

  /*! Returns the number of fields */
  int n_fields() const { return static_cast<int>(name_id_bimap_.size()); }

  /*! Returns the number of fields of a given category */
  int n_fields(field::Kind kind) const { return get_id_range(kind).n_items(); }

  /*! \brief Returns the number of fields associated with a the specified pack_id */
  int n_fields(uint8_t pack_id) const
  {
    if (pack_id < n_packs()) {
      return static_cast<int>(pack_info_[pack_id].flat_idx_slc.size());
    }
    return 0;
  }

  /*! \brief Returns the number of field-packs */
  int n_packs() const { return pname_id_bimap_.size(); }

  /*! This returns a "range" over all ids
   *
   *  This might be used in a case like the following:
   *  \code{c++}
   *  for (FieldId field_id: field_info.get_id_range(field::Kind::HYDRO)) {
   *    // ...
   *  }
   *  \endcode
   */
  field::IdRange get_id_range(field::Kind kind) const;
};