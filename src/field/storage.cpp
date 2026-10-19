/*! \file
 *  Implement the \ref field_detail::Storage type.
 */

#include "storage.h"

#include <utility>  // std::move

#include "../utils/error_handling.h"
#include "field_info.h"

namespace field_detail
{

Storage::Storage(const FieldInfo& info, std::array<int, 3> shape_xyz)
{
  // we do some dumb stuff right now
  // (i.e. I suspect we'll end up using a builder pattern once we start having multiple
  // packs of fields)

  // error checks:
  if (shape_xyz[0] == 1) {
    CHOLLA_ERROR("x-axis extent can never be 1");
  } else if ((shape_xyz[1] == 1) and (shape_xyz[2] != 1)) {
    CHOLLA_ERROR("y-axis extent can't be 1 when z-axis extent isn't 1");
  }
  const char axes[] = "xyz";
  for (int i = 0; i < 3; i++) {
    CHOLLA_ASSERT(shape_xyz[i] > 0, "%c-axis extent must be positive", axes[i]);
  }

  int tmp = info.n_packs();
  CHOLLA_ASSERT(tmp <= UINT8_MAX, "something went wrong!");
  uint8_t n_packs = static_cast<uint8_t>(tmp);
  for (uint8_t pack_id = 0; pack_id < n_packs; pack_id++) {
    std::optional<int> maybe_n_dev_registers  = info.n_registers_from_pack_id(pack_id, MemSpace::DEV);
    std::optional<int> maybe_n_host_registers = info.n_registers_from_pack_id(pack_id, MemSpace::HOST);
    std::optional<int> maybe_ghost_depth      = info.ghost_depth_from_pack_id(pack_id);
    if (!maybe_n_dev_registers.has_value() || !maybe_n_host_registers.has_value()) {
      CHOLLA_ERROR("problem number of registers");
    } else if (!maybe_ghost_depth.has_value()) {
      CHOLLA_ERROR("problem accessing the ghost depth of the pack");
    }
    int n_dev_registers  = maybe_n_dev_registers.value();
    int n_host_registers = maybe_n_host_registers.value();
    int cur_ghost_depth  = maybe_ghost_depth.value();

    if ((n_dev_registers < 0) || (n_host_registers < 0)) {
      CHOLLA_ERROR("can't have negative numbers of registers");
    } else if ((n_dev_registers > static_cast<int>(field_detail::MAX_REGISTERS)) ||
               (n_host_registers > static_cast<int>(field_detail::MAX_REGISTERS))) {
      CHOLLA_ERROR("too many registers");
    } else if ((n_dev_registers == 0) && (n_host_registers == 0)) {
      CHOLLA_ERROR("a pack needs at least 1 host or device register");
    }
    // let's compute slots_stride (you can think of this as elements per field)
    std::size_t elem_per_field = 1;
    for (const int& v : shape_xyz) {
      elem_per_field *= static_cast<std::size_t>((v == 1) ? v : v + 2 * cur_ghost_depth);
    }

    PackData tmp;
    tmp.slot_stride               = elem_per_field;
    std::size_t elements_per_pack = info.n_fields(pack_id) * elem_per_field;
    for (int i = 0; i < n_host_registers; i++) {
      // this is unfortunately a little clunky
      Real* host_ptr = nullptr;
      GPU_Error_Check(cudaHostAlloc((void**)&host_ptr, elements_per_pack * sizeof(Real), cudaHostAllocDefault));
      tmp.host_registers[i] = std::unique_ptr<Real, CudaHostPtrDelete>(host_ptr, CudaHostPtrDelete{});
    }

    for (int i = 0; i < n_dev_registers; i++) {
      // a register on device is simpler since its represented by a DeviceVector
      tmp.dev_registers[i].resize(elements_per_pack);
    }
    pack_vec.emplace_back(std::move(tmp));
  }
}

}  // namespace field_detail