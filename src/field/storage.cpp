/*! \file
 *  Implement the \ref field_detail::Storage type.
 */

#include "storage.h"

#include <utility>  // std::move

#include "../utils/error_handling.h"
#include "field_info.h"

namespace field_detail
{

Storage::Storage(const FieldInfo& info, std::array<int, 3> shape_xyz, int ghost_depth)
{
  // we do some dumb stuff right now
  // (i.e. I suspect we'll end up using a builder pattern once we start having multiple
  // packs of fields)

  const int n_registers = 1;
  // error checks:
  CHOLLA_ASSERT(ghost_depth >= 0, "ghost_depth can't be negative");
  if (shape_xyz[0] == 1) {
    CHOLLA_ERROR("x-axis extent can never be 1");
  } else if ((shape_xyz[1] == 1) and (shape_xyz[2] != 1)) {
    CHOLLA_ERROR("y-axis extent can't be 1 when z-axis extent isn't 1");
  }
  const char axes[] = "xyz";
  for (int i = 0; i < 3; i++) {
    CHOLLA_ASSERT(shape_xyz[i] > 0, "%c-axis extent must be positive", axes[i]);
  }

  int n_packs = info.n_packs();
  for (uint8_t pack_id = 0; pack_id < n_packs; pack_id++) {
    // in the future, n_registers & cur_ghost_depth may to vary between packs
    int n_registers     = 1;
    int cur_ghost_depth = ghost_depth;

    // let's compute slots_stride (you can think of this as elements per field)
    std::size_t elem_per_field = 1;
    for (const int& v : shape_xyz) {
      elem_per_field *= static_cast<std::size_t>((v == 1) ? v : v + 2 * cur_ghost_depth);
    }

    PackData tmp;
    tmp.slot_stride               = elem_per_field;
    std::size_t elements_per_pack = info.n_fields(pack_id) * elem_per_field;
    for (int i = 0; i < n_registers; i++) {
      // this is unfortunately a little clunky
      Real* host_ptr = nullptr;
      GPU_Error_Check(cudaHostAlloc((void**)&host_ptr, elements_per_pack * sizeof(Real), cudaHostAllocDefault));
      tmp.host_registers[i] = std::unique_ptr<Real, CudaHostPtrDelete>(host_ptr, CudaHostPtrDelete{});

      // a register on device is simpler since its represented by a DeviceVector
      tmp.dev_registers[i].resize(elements_per_pack);
    }
    pack_vec.emplace_back(std::move(tmp));
  }
}

std::optional<const Real*> Storage::field(MemSpace s, FieldId id, std::size_t register_idx) const
{
  if (id.pack_id >= pack_vec.size()) {
    return std::nullopt;
  }
  const PackData& pack_data = pack_vec.at(id.pack_id);
  std::size_t offset        = pack_data.slot_stride * static_cast<std::size_t>(id.slot_idx);
  if (offset >= pack_data.elements_per_pack_register()) {
    return std::nullopt;
  }
  if (s == MemSpace::HOST) {
    return {pack_data.host_registers[register_idx].get() + offset};
  } else {
    return {pack_data.dev_registers[register_idx].data() + offset};
  }
}

}  // namespace field_detail