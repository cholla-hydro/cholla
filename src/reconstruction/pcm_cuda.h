/*! \file pcm_cuda.h
 *  \brief Declarations of the pcm function */

#ifndef PCM_CUDA_H
#define PCM_CUDA_H

#include "../utils/basic_structs.h"
#include "../utils/hydro_utilities.h"

namespace reconstruction
{
template <size_t direction>
reconstruction::InterfaceState __device__ __host__ inline PCM_Reconstruction(Real const *dev_conserved,
                                                                             size_t const xid, size_t const yid,
                                                                             size_t const zid, size_t const nx,
                                                                             size_t const ny, size_t const n_cells,
                                                                             Real const gamma)
{
  // Load cell conserved
  hydro_utilities::Conserved conserved_data =
      hydro_utilities::Load_Cell_Conserved<direction>(dev_conserved, xid, yid, zid, nx, ny, n_cells);

  // Convert cell to primitives
  hydro_utilities::Primitive primitive_data = hydro_utilities::Conserved_2_Primitive(conserved_data, gamma);

  // Integrate cell values into an InterfaceState
  reconstruction::InterfaceState interface_state;

  interface_state.density  = conserved_data.density;
  interface_state.velocity = primitive_data.velocity;
  interface_state.momentum = conserved_data.momentum;
  interface_state.pressure = primitive_data.pressure;
  interface_state.energy   = conserved_data.energy;

#ifdef MHD
  interface_state.total_pressure = mhd::utils::computeTotalPressure(primitive_data.pressure, conserved_data.magnetic);
  interface_state.magnetic       = conserved_data.magnetic;
#endif  // MHD
#ifdef DE
  interface_state.gas_energy_specific = primitive_data.gas_energy_specific;
#endif  // DE
#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    interface_state.scalar_specific[i] = primitive_data.scalar_specific[i];
  }
#endif  // SCALAR

  return interface_state;
}
}  // namespace reconstruction
#endif  // PCM_CUDA_H
