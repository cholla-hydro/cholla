/*! \file
 *  Implement the \ref FieldManager type.
 */

#include "field_manager.h"

#include <sstream>
#include <string>
#include <type_traits>

#include "../utils/error_handling.h"

// a helper function intended to produce an informative error message for when either
// FieldManager::field_or_abort or FieldManager::pack_or_abort fails
template <typename T>
static std::string infer_abort_msg_(MemSpace s, const T& id, std::size_t register_idx, const FieldInfo& info)
{
  const char* data_type = nullptr;
  std::optional<std::string> maybe_name;
  if constexpr (std::is_same_v<T, FieldId>) {
    data_type  = "field";
    maybe_name = info.field_name(id);
  } else {
    data_type  = "field-pack";
    maybe_name = info.pack_name(id);
  }

  if (!maybe_name.has_value()) {
    return std::string("the specified id doesn't correspond to a known ") + data_type;
  }

  const char* space_name = (s == MemSpace::DEV) ? "device" : "host";
  std::ostringstream sstream;
  sstream << "The specified id, corresponds to the \"" << maybe_name.value() << "\" " << data_type
          << ". Cholla is NOT configured to track a " << space_name << " register (with register_idx=" << register_idx
          << ") for that " << data_type;
  return sstream.str();
}

bool FieldManager::pack_has_register_(MemSpace s, uint8_t pack_id, std::size_t register_idx) const
{
  std::optional<int> maybe_n_registers = field_info_.n_registers_from_pack_id(pack_id, s);
  return maybe_n_registers.value_or(0) > static_cast<int>(register_idx);
}

std::optional<Real*> FieldManager::field(MemSpace s, FieldId id, std::size_t register_idx)
{
  if (!field_has_register_(s, id, register_idx)) return std::nullopt;
  return storage_.field(s, id, register_idx);
}

std::optional<const Real*> FieldManager::field(MemSpace s, FieldId id, std::size_t register_idx) const
{
  if (!field_has_register_(s, id, register_idx)) return std::nullopt;
  return storage_.field(s, id, register_idx);
}

std::optional<Real*> FieldManager::field(MemSpace s, std::string_view name, std::size_t register_idx)
{
  std::optional<FieldId> maybe_id = field_info_.field_id(name);
  if (maybe_id.has_value()) {
    if (!field_has_register_(s, *maybe_id, register_idx)) return std::nullopt;
    return storage_.field(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

std::optional<const Real*> FieldManager::field(MemSpace s, std::string_view name, std::size_t register_idx) const
{
  std::optional<FieldId> maybe_id = field_info_.field_id(name);
  if (maybe_id.has_value()) {
    if (!field_has_register_(s, *maybe_id, register_idx)) return std::nullopt;
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
  std::string msg = infer_abort_msg_(s, id, register_idx, field_info_);
  CHOLLA_ERROR("%s", msg.c_str());
}

const Real* FieldManager::field_or_abort(MemSpace s, FieldId id, std::size_t register_idx) const
{
  std::optional<const Real*> maybe_ptr = field(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  std::string msg = infer_abort_msg_(s, id, register_idx, field_info_);
  CHOLLA_ERROR("%s", msg.c_str());
}

std::optional<Real*> FieldManager::pack(MemSpace s, uint8_t id, std::size_t register_idx)
{
  if (!pack_has_register_(s, id, register_idx)) return std::nullopt;
  return storage_.pack(s, id, register_idx);
}

std::optional<const Real*> FieldManager::pack(MemSpace s, uint8_t id, std::size_t register_idx) const
{
  if (!pack_has_register_(s, id, register_idx)) return std::nullopt;
  return storage_.pack(s, id, register_idx);
}

std::optional<Real*> FieldManager::pack(MemSpace s, std::string_view name, std::size_t register_idx)
{
  std::optional<uint8_t> maybe_id = field_info_.pack_id(name);
  if (maybe_id.has_value()) {
    if (!pack_has_register_(s, *maybe_id, register_idx)) return std::nullopt;
    return storage_.pack(s, *maybe_id, register_idx);
  }
  return std::nullopt;
}

std::optional<const Real*> FieldManager::pack(MemSpace s, std::string_view name, std::size_t register_idx) const
{
  std::optional<uint8_t> maybe_id = field_info_.pack_id(name);
  if (maybe_id.has_value()) {
    if (!pack_has_register_(s, *maybe_id, register_idx)) return std::nullopt;
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
  std::string msg = infer_abort_msg_(s, id, register_idx, field_info_);
  CHOLLA_ERROR("%s", msg.c_str());
}

const Real* FieldManager::pack_or_abort(MemSpace s, uint8_t id, std::size_t register_idx) const
{
  std::optional<const Real*> maybe_ptr = pack(s, id, register_idx);
  if (maybe_ptr.has_value()) {
    return *maybe_ptr;
  }
  std::string msg = infer_abort_msg_(s, id, register_idx, field_info_);
  CHOLLA_ERROR("%s", msg.c_str());
}