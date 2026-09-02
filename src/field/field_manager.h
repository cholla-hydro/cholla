/*! \file
 *  Declare the \ref FieldManager type.
 */
#pragma once

#include "../utils/error_handling.h"
#include "field_info.h"
#include "storage.h"

/*! \brief Manages field data
 *
 *  Implementation Note:
 *  - this object is composed from a \ref FieldInfo object & a \ref field_info::Storage
 *    object.
 *  - It's worth considering that it makes some sense to support the existence of a
 *    \ref FieldInfo object outside of the object.
 *  - in contrast, the \ref field_info::Storage object is an implementation detail
 */
class FieldManager
{
  // the order of defining these data-members actually matters for the purposes of
  // initialization

  /// tracks field data
  field_detail::Storage storage_;

  /// tracks field information
  FieldInfo field_info_;

 public:
  /*! \brief Construct a new instance
   *
   *  \note
   *  This is intentionally very simple right now. See the constructor of
   *  \ref field_detail::Storage for details.
   */
  FieldManager(FieldInfo info, std::array<int, 3> shape_xyz, int ghost_depth)
      // reminder: the order that data members are declared in is quite important
      : storage_(info, shape_xyz, ghost_depth), field_info_(std::move(info))
  {
  }

  ///@{
  /*! \brief try to retrieve the specified field
   *
   *  \note
   *  The ``or_abort`` variant is only defined for the ``FieldId`` argument since the
   *  caller presumably already looked up the ``FieldId``.
   */
  std::optional<Real*> field(MemSpace s, FieldId id, std::size_t register_idx = 0);
  std::optional<const Real*> field(MemSpace s, FieldId id, std::size_t register_idx = 0) const;
  std::optional<Real*> field(MemSpace s, std::string_view name, std::size_t register_idx = 0);
  std::optional<const Real*> field(MemSpace s, std::string_view name, std::size_t register_idx = 0) const;
  Real* field_or_abort(MemSpace s, FieldId id, std::size_t register_idx = 0);
  const Real* field_or_abort(MemSpace s, FieldId id, std::size_t register_idx = 0) const;
  ///@}

  ///@{
  /*! \brief try to retrieve the specified field pack
   *
   *  \note
   *  The ``or_abort`` variant is only defined for the ``id`` argument since the caller
   *  presumably already looked up the ``id``.
   */
  std::optional<Real*> pack(MemSpace s, uint8_t id, std::size_t register_idx = 0);
  std::optional<const Real*> pack(MemSpace s, uint8_t id, std::size_t register_idx = 0) const;
  std::optional<Real*> pack(MemSpace s, std::string_view name, std::size_t register_idx = 0);
  std::optional<const Real*> pack(MemSpace s, std::string_view name, std::size_t register_idx = 0) const;
  Real* pack_or_abort(MemSpace s, uint8_t id, std::size_t register_idx = 0);
  const Real* pack_or_abort(MemSpace s, uint8_t id, std::size_t register_idx = 0) const;
  ///@}

  /*! \brief */

  /*! \brief swap the field_pack pointers in \p reg_0 and \p reg_1 */
  void swap_registers(MemSpace s, uint8_t pack_id, std::size_t reg_0, std::size_t reg_1)
  {
    // in order to implement this, we'll need to do a little work registering some
    // callback functions
    CHOLLA_ERROR("NOT IMPLEMENTED YET");
  }

  /*! \brief retrieve the field info object */
  const FieldInfo& info() const { return field_info_; }

  // a bunch of convenience functions
  // -> we currently don't bother implementing functions to query whether a field is a
  //    part of a field-pack or what the slot of a field in a field-pack. In the primary
  //    case that we care about (i.e. mhd fields) the field-packs are constructed to
  //    mirror hard-coded expectations
  // -> obviously we can adjust these going forward, but people are also free to access
  //    the FieldInfo object directly

  /*! \brief try to lookup the field_id associated with the field_name */
  std::optional<FieldId> field_id(std::string_view name) const { return field_info_.lookup_FieldID(name); }

  /*! \brief try to lookup the associated pack_id */
  std::optional<uint8_t> pack_id(std::string_view pack_name) const { return field_info_.pack_id(pack_name); }

  /*! \brief try to look up the field name */
  std::optional<std::string> field_name(FieldId id) const { return field_info_.field_name(id); }

  /*! \brief try to look up the pack name */
  std::optional<std::string> pack_name(uint8_t id) const { return field_info_.pack_name(id); }
};