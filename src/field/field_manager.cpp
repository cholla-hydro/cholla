/*! \file
 *  Implement the \ref FieldManager type.
 */

#include "field_manager.h"

#include "../utils/error_handling.h"

std::optional<Real*> FieldManager::field(MemSpace s, FieldId id, std::size_t register_idx)
{
  return storage_.field(s, id, register_idx);
}

std::optional<const Real*> FieldManager::field(MemSpace s, FieldId id, std::size_t register_idx) const
{
  return storage_.field(s, id, register_idx);
}

std::optional<Real*> FieldManager::field(MemSpace s, std::string_view name, std::size_t register_idx)
{
  std::optional<FieldId> maybe_id = field_info_.lookup_FieldID(name);
  if (maybe_id.has_value()) {
    return storage_.field(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

std::optional<const Real*> FieldManager::field(MemSpace s, std::string_view name, std::size_t register_idx) const
{
  std::optional<FieldId> maybe_id = field_info_.lookup_FieldID(name);
  if (maybe_id.has_value()) {
    return storage_.field(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

Real* FieldManager::field_or_abort(MemSpace s, FieldId id, std::size_t register_idx)
{
  std::optional<Real*> maybe_ptr = field(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  CHOLLA_ERROR("unable to find field associated with field_id");
}

const Real* FieldManager::field_or_abort(MemSpace s, FieldId id, std::size_t register_idx) const
{
  std::optional<const Real*> maybe_ptr = field(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  CHOLLA_ERROR("unable to find field associated with field_id");
}

std::optional<Real*> FieldManager::pack(MemSpace s, uint8_t id, std::size_t register_idx)
{
  return storage_.pack(s, id, register_idx);
}

std::optional<const Real*> FieldManager::pack(MemSpace s, uint8_t id, std::size_t register_idx) const
{
  return storage_.pack(s, id, register_idx);
}

std::optional<Real*> FieldManager::pack(MemSpace s, std::string_view name, std::size_t register_idx)
{
  std::optional<uint8_t> maybe_id = field_info_.pack_id(name);
  if (maybe_id.has_value()) {
    return storage_.pack(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

std::optional<const Real*> FieldManager::pack(MemSpace s, std::string_view name, std::size_t register_idx) const
{
  std::optional<uint8_t> maybe_id = field_info_.pack_id(name);
  if (maybe_id.has_value()) {
    return storage_.pack(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

Real* FieldManager::pack_or_abort(MemSpace s, uint8_t id, std::size_t register_idx)
{
  std::optional<Real*> maybe_ptr = pack(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  CHOLLA_ERROR("unable to find pack associated with pack_id");
}

const Real* FieldManager::pack_or_abort(MemSpace s, uint8_t id, std::size_t register_idx) const
{
  std::optional<const Real*> maybe_ptr = pack(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  CHOLLA_ERROR("unable to find pack associated with pack_id");
}