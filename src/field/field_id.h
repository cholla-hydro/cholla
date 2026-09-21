/*! \file
 *  Declare/implement the \ref FieldId type.
 */

#pragma once

#include <cstdint>  // uint8_t

// forward declarations
class FieldInfo;
class FieldManager;
namespace field_detail
{
class Storage;
}  // namespace field_detail

/*! \brief Represents a field identifier
 *
 *  Code outside of the field machinery should always treat this like the
 *  internals are entirely private (we expose the internals to other field
 *  machinery because there are some optimization advantages to using an
 *  aggregate)
 */
struct FieldId {
  // when the NDEBUG macro isn't defined (the standard way to indicate that
  // the program is being compiled in debug-mode), we explicitly make the
  // internal contents private, so that code outside of the field machinery
  // doesn't accidentally access this machinery
#ifndef NDEBUG
  friend FieldInfo;
  friend FieldManager;
  friend field_detail::Storage;

 private:
#endif

  /// The id of the field-pack
  uint8_t pack_id;

  /// The slot index within the field-pack
  uint8_t slot_idx;

  /*! \brief Primary constructor */
  FieldId(uint8_t pack_id, uint8_t slot_idx) noexcept : pack_id{pack_id}, slot_idx{slot_idx} {}

 public:
  /*! \brief Default constructor (initialized in an intentionally invalid state) */
  FieldId() noexcept : pack_id{UINT8_MAX}, slot_idx{UINT8_MAX} {}

  FieldId(const FieldId&)            = default;
  FieldId(FieldId&&)                 = default;
  FieldId& operator=(const FieldId&) = default;
  FieldId& operator=(FieldId&&)      = default;

  bool operator==(const FieldId& o) { return (pack_id == o.pack_id) && (slot_idx == o.slot_idx); }
  bool operator!=(const FieldId& o) { return (pack_id != o.pack_id) || (slot_idx != o.slot_idx); }
};