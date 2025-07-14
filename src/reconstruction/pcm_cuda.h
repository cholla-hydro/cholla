/*! \file pcm_cuda.h
 *  \brief Declarations of the pcm function */

#ifndef PCM_CUDA_H
#define PCM_CUDA_H

#include "../utils/basic_structs.h"
#include "../utils/hydro_utilities.h"

__global__ void PCM_Reconstruction_1D(Real *dev_conserved, Real *dev_bounds_L, Real *dev_bounds_R, int n_cells,
                                      int n_ghost, Real gamma, int n_fields);

__global__ void PCM_Reconstruction_2D(Real *dev_conserved, Real *dev_bounds_Lx, Real *dev_bounds_Rx,
                                      Real *dev_bounds_Ly, Real *dev_bounds_Ry, int nx, int ny, int n_ghost, Real gamma,
                                      int n_fields);

__global__ void PCM_Reconstruction_3D(Real *dev_conserved, Real *dev_bounds_Lx, Real *dev_bounds_Rx,
                                      Real *dev_bounds_Ly, Real *dev_bounds_Ry, Real *dev_bounds_Lz,
                                      Real *dev_bounds_Rz, int nx, int ny, int nz, int n_ghost, Real gamma,
                                      int n_fields);

namespace reconstruction
{
/*!
 * \brief Perform PCM reconstruction for a given cell
 *
 * \tparam direction The direction that the solve is taking place in. 0=X, 1=Y, 2=Z
 * \param[in] dev_conserved The converved variable array
 * \param[in] xid The x-direction cell id
 * \param[in] yid The y-direction cell id
 * \param[in] zid The z-direction cell id
 * \param[in] nx The number of cells in the x-direction
 * \param[in] ny The number of cells in the y-direction
 * \param[in] n_cells The total number of cells
 * \param[in] gamma The adiabatic index
 * \return reconstruction::InterfaceState The interface state at xid, yid, zid
 */
template <uint direction>
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
  interface_state.gas_energy = primitive_data.gas_energy;
#endif  // DE
#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    interface_state.scalar[i] = primitive_data.scalar[i];
  }
#endif  // SCALAR

  return interface_state;
}
}  // namespace reconstruction
#endif  // PCM_CUDA_H
