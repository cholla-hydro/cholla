/*! \file
 *  Declare the \ref field_detail::Storage type.
 */
#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include "../global/global.h"
#include "../utils/DeviceVector.h"
#include "../utils/error_handling.h"
#include "field_id.h"

// forward declarations
class FieldInfo;

/*! \brief Represents a memory space */
enum class MemSpace { HOST, DEV };

/*! \brief Holds implementation details pertaining to field data */
namespace field_detail
{

/*! Specifies the maximum number of field pack registers.
 *
 *  \note Some day, this might need to be a bigger number.
 */
inline constexpr std::size_t MAX_REGISTERS = 2;

/*! \brief Deletion policy used by a smart pointer that manages a pointer in host
 *      memory allocated via ``cudaHostAlloc`` */
struct CudaHostPtrDelete {
  template <class T>
  void operator()(T* ptr) const
  {
    if (ptr == nullptr) return;
    GPU_Error_Check(cudaFreeHost(static_cast<void*>(ptr)));
  }
};

/*! tracks all memory associated with a single field-pack.
 *
 *  The @ref FieldInfo object has the responsibility of tracking the number of registers
 *  associated with a field pack.
 *
 *  \note
 *  Unused registers simply hold empty containers.
 *
 *  Implementation Strategy
 *  =======================
 *  The current implementation creates separate allocations for each register. A more
 *  robust strategy **might** be to make a single allocation for all registers and
 *  tracking the pointer offsets to the start of the register.
 */
struct PackData {
  /// The stride for accessing each slot in a field pack
  std::size_t slot_stride;
  /// Holds register-data in host-memory
  std::array<std::unique_ptr<Real, CudaHostPtrDelete>, MAX_REGISTERS> host_registers;
  /// Holds register-data in device-memory
  std::array<cuda_utilities::DeviceVector<Real>, MAX_REGISTERS> dev_registers;

  std::size_t elements_per_pack_register() const { return dev_registers[0].size(); }
};

/*! tracks all memory associated across all field data
 *
 *  This is intentionally bare-bones. Most of the details about the fields
 *  (e.g. their names, the names of packs, the number of fields per pack)
 *  is tracked separately.
 */
class Storage
{
  // An important invariant: no constructor or assignment operations
  // should occur after this is constructed. This is important for
  // avoiding dangling pointers
  std::vector<PackData> pack_vec;

  /*! \brief Helper function that implements the \ref field method.
   *
   *  This is a template that properly deduces whether to return a
   *  ``std::optional<const Real*>`` or ``std::optional<Real*>`` based on the constness
   *  of the first argument
   *
   *  \note
   *  This was written to avoid a const-cast based on a recommendation from the
   *  [C++ core guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es50-dont-cast-away-const)
   */
  template <class T>
  static auto field_impl_(T& storage, MemSpace s, FieldId id, std::size_t register_idx)
  {
    using Ptr = std::conditional_t<std::is_const_v<T>, const Real*, Real*>;
    if (id.pack_id >= storage.pack_vec.size()) {
      return std::optional<Ptr>();  // an empty optional
    }

    // pack_data has a type of `const PackData&` or `PackData&`
    auto& pack_data    = storage.pack_vec.at(id.pack_id);
    std::size_t offset = pack_data.slot_stride * static_cast<std::size_t>(id.slot_idx);
    if (offset >= pack_data.elements_per_pack_register()) {
      return std::optional<Ptr>();  // an empty optional
    }

    Ptr ptr;
    if (s == MemSpace::HOST) {
      ptr = pack_data.host_registers[register_idx].get() + offset;
    } else {
      ptr = pack_data.dev_registers[register_idx].data() + offset;
    }
    return std::optional<Ptr>(ptr);
  }

  /*! \brief Helper function that implements the \ref pack method. */
  template <class T>
  static auto pack_impl_(T& storage, MemSpace s, std::size_t id, std::size_t register_idx)
  {
    using Ptr = std::conditional_t<std::is_const_v<T>, const Real*, Real*>;
    if (id > storage.pack_vec.size()) {
      return std::optional<Ptr>();  // an empty optional
    } else {
      FieldId field_id(static_cast<uint8_t>(id), 0);
      return field_impl_(storage, s, field_id, register_idx);
    }
  }

 public:
  /*! \brief Construct a new instance
   *
   *  \note
   *  This is intentionally very simple right now. In the future, different packs of
   *  fields may have different numbers of ghost zones (and ideally we would allow the
   *  number of ghost zones to vary with direction)
   */
  Storage(const FieldInfo& info, std::array<int, 3> shape_xyz, int ghost_depth);

  // we use default definitions for move construction/assingment
  Storage(Storage&&)            = default;
  Storage& operator=(Storage&&) = default;

  // delete copy constructor and copy assignment (we can add it back later, but
  // in the current implementation, it's always a mistake to invoke them)
  Storage(const Storage&)            = delete;
  Storage& operator=(const Storage&) = delete;

  /*! \brief Retrieve the specified field
   *
   *  The returned optional is either empty or it holds a non-null pointer
   */
  std::optional<Real*> field(MemSpace s, FieldId id, std::size_t register_idx = 0)
  {
    return field_impl_(*this, s, id, register_idx);
  }

  std::optional<const Real*> field(MemSpace s, FieldId id, std::size_t register_idx = 0) const
  {
    return field_impl_(*this, s, id, register_idx);
  }

  /*! \brief retrieve the specified field pack
   *
   *  The returned optional is either empty or it holds a non-null pointer
   */
  std::optional<Real*> pack(MemSpace s, std::size_t id, std::size_t register_idx = 0)
  {
    return pack_impl_(*this, s, id, register_idx);
  }

  std::optional<const Real*> pack(MemSpace s, std::size_t id, std::size_t register_idx = 0) const
  {
    return pack_impl_(*this, s, id, register_idx);
  }

  /*! \brief swap the field_pack pointers in \p reg_0 and \p reg_1 */
  void swap_registers(MemSpace s, std::size_t pack_id, std::size_t reg_0, std::size_t reg_1)
  {
    PackData& pack_data = pack_vec.at(pack_id);

    if (s == MemSpace::HOST) {
      pack_data.host_registers[reg_0].swap(pack_data.host_registers[reg_1]);
    } else {
      pack_data.dev_registers[reg_0].swap(pack_data.dev_registers[reg_1]);
    }
  }
};

}  // namespace field_detail