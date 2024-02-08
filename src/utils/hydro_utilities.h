/*!
 * \file hydro_utilities.h
 * \author Helena Richie (helenarichie@pitt.edu)
 * \brief Contains the declaration of various utility functions for hydro
 *
 */

#pragma once

#include <iostream>
#include <string>

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/basic_structs.h"
#include "../utils/gpu.hpp"
#include "../utils/math_utilities.h"
#include "../utils/mhd_utilities.h"

/*!
 * INDEX OF VARIABLES
 * P : pressure
 * vx, vy, vz : x, y, and z velocity
 * d : density
 * E : energy
 * T : temperature
 * mx, my, mz : x, y, and z momentum
 * n : number density
 */

namespace hydro_utilities
{
inline __host__ __device__ Real Calc_Pressure_Primitive(Real const &E, Real const &d, Real const &vx, Real const &vy,
                                                        Real const &vz, Real const &gamma, Real const &magnetic_x = 0.0,
                                                        Real const &magnetic_y = 0.0, Real const &magnetic_z = 0.0)
{
  Real pressure = E - 0.5 * d * math_utils::SquareMagnitude(vx, vy, vz);

#ifdef MHD
  pressure -= mhd::utils::computeMagneticEnergy(magnetic_x, magnetic_y, magnetic_z);
#endif  // MHD

  return fmax((gamma - 1.) * pressure, TINY_NUMBER);
}

inline __host__ __device__ Real Calc_Pressure_Conserved(Real const &E, Real const &d, Real const &mx, Real const &my,
                                                        Real const &mz, Real const &gamma, Real const &magnetic_x = 0.0,
                                                        Real const &magnetic_y = 0.0, Real const &magnetic_z = 0.0)
{
  Real pressure = E - 0.5 * math_utils::SquareMagnitude(mx, my, mz) / d;

#ifdef MHD
  pressure -= mhd::utils::computeMagneticEnergy(magnetic_x, magnetic_y, magnetic_z);
#endif  // MHD

  return fmax((gamma - 1.) * pressure, TINY_NUMBER);
}

inline __host__ __device__ Real Calc_Temp(Real const &P, Real const &n)
{
  Real T = P * PRESSURE_UNIT / (n * KB);
  return T;
}

/*!
 * \brief Compute the temperature from the conserved variables
 *
 * \param[in] E The energy
 * \param[in] d The density
 * \param[in] mx The momentum in the X-direction
 * \param[in] my The momentum in the Y-direction
 * \param[in] mz The momentum in the Z-direction
 * \param[in] gamma The adiabatic index
 * \param[in] n The number density
 * \param[in] magnetic_x The cell centered magnetic field in the X-direction
 * \param[in] magnetic_y The cell centered magnetic field in the Y-direction
 * \param[in] magnetic_z The cell centered magnetic field in the Z-direction
 * \return Real The temperature of the gas in a cell
 */
inline __host__ __device__ Real Calc_Temp_Conserved(Real const E, Real const d, Real const mx, Real const my,
                                                    Real const mz, Real const gamma, Real const n,
                                                    Real const magnetic_x = 0.0, Real const magnetic_y = 0.0,
                                                    Real const magnetic_z = 0.0)
{
  Real const P = Calc_Pressure_Conserved(E, d, mx, my, mz, gamma, magnetic_x, magnetic_y, magnetic_z);
  return Calc_Temp(P, n);
}

#ifdef DE
/*!
 * \brief Compute the temperature when DE is turned on
 *
 * \param[in] gas_energy The total gas energy in the cell. This is the value stored in the grid at
 * grid_enum::GasEnergy
 * \param[in] gamma The adiabatic index
 * \param[in] n The number density
 * \return Real The temperature
 */
inline __host__ __device__ Real Calc_Temp_DE(Real const gas_energy, Real const gamma, Real const n)
{
  return gas_energy * (gamma - 1.0) * PRESSURE_UNIT / (n * KB);
}
#endif  // DE

inline __host__ __device__ Real Calc_Energy_Primitive(Real const &P, Real const &d, Real const &vx, Real const &vy,
                                                      Real const &vz, Real const &gamma, Real const &magnetic_x = 0.0,
                                                      Real const &magnetic_y = 0.0, Real const &magnetic_z = 0.0)
{
  // Compute and return energy
  Real energy = (fmax(P, TINY_NUMBER) / (gamma - 1.)) + 0.5 * d * math_utils::SquareMagnitude(vx, vy, vz);

#ifdef MHD
  energy += mhd::utils::computeMagneticEnergy(magnetic_x, magnetic_y, magnetic_z);
#endif  // MHD

  return energy;
}

inline __host__ __device__ Real Calc_Energy_Conserved(Real const &P, Real const &d, Real const &momentum_x,
                                                      Real const &momentum_y, Real const &momentum_z, Real const &gamma,
                                                      Real const &magnetic_x = 0.0, Real const &magnetic_y = 0.0,
                                                      Real const &magnetic_z = 0.0)
{
  // Compute and return energy
  Real energy = (fmax(P, TINY_NUMBER) / (gamma - 1.)) +
                (0.5 / d) * math_utils::SquareMagnitude(momentum_x, momentum_y, momentum_z);

#ifdef MHD
  energy += mhd::utils::computeMagneticEnergy(magnetic_x, magnetic_y, magnetic_z);
#endif  // MHD

  return energy;
}

inline __host__ __device__ Real Get_Pressure_From_DE(Real const &E, Real const &U_total, Real const &U_advected,
                                                     Real const &gamma)
{
  Real U, P;
  Real eta = DE_ETA_1;
  // Apply same condition as Byan+2013 to select the internal energy from which
  // compute pressure.
  if (U_total / E > eta) {
    U = U_total;
  } else {
    U = U_advected;
  }
  P = U * (gamma - 1.0);
  return fmax(P, (Real)TINY_NUMBER);
  ;
}

/*!
 * \brief Compute the kinetic energy from the density and velocities
 *
 * \param[in] d The density
 * \param[in] vx The x velocity
 * \param[in] vy The y velocity
 * \param[in] vz The z velocity
 * \return Real The kinetic energy
 */
inline __host__ __device__ Real Calc_Kinetic_Energy_From_Velocity(Real const &d, Real const &vx, Real const &vy,
                                                                  Real const &vz)
{
  return 0.5 * d * math_utils::SquareMagnitude(vx, vy, vz);
}

/*!
 * \brief Compute the kinetic energy from the density and momenta
 *
 * \param[in] d The density
 * \param[in] mx The x momentum
 * \param[in] my The y momentum
 * \param[in] mz The z momentum
 * \return Real The kinetic energy
 */
inline __host__ __device__ Real Calc_Kinetic_Energy_From_Momentum(Real const &d, Real const &mx, Real const &my,
                                                                  Real const &mz)
{
  return (0.5 / d) * math_utils::SquareMagnitude(mx, my, mz);
}

/*!
 * \brief Compute the sound speed in the cell from conserved variables
 *
 * \param E Energy
 * \param d densidy
 * \param mx x momentum
 * \param my y momentum
 * \param mz z momentum
 * \param gamma adiabatic index
 * \return Real The sound speed
 */
inline __host__ __device__ Real Calc_Sound_Speed(Real const &E, Real const &d, Real const &mx, Real const &my,
                                                 Real const &mz, Real const &gamma)
{
  Real P = Calc_Pressure_Conserved(E, d, mx, my, mz, gamma);
  return sqrt(gamma * P / d);
}

/*!
 * \brief Compute the sound in the cell from primitive variables
 *
 * \param P
 * \param d
 * \param gamma
 * \return __host__
 */
inline __host__ __device__ Real Calc_Sound_Speed(Real const &P, Real const &d, Real const &gamma)
{
  return sqrt(gamma * P / d);
}

// =====================================================================================================================
template <size_t dir = 0>
inline __host__ __device__ Conserved Load_Cell_Conserved(Real const *dev_conserved, size_t const xid, size_t const yid,
                                                         size_t const zid, size_t const nx, size_t const ny,
                                                         size_t const n_cells)
{
  // First, check that our direction is correct
  static_assert((0 <= dir) and (dir <= 2), "dir is not in the proper range");

  // Compute index
  size_t const cell_id = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);

  // Load all the data
  Conserved loaded_data;

  // Hydro variables
  loaded_data.density    = dev_conserved[cell_id + n_cells * grid_enum::density];
  loaded_data.momentum.x = dev_conserved[cell_id + n_cells * grid_enum::momentum_x];
  loaded_data.momentum.y = dev_conserved[cell_id + n_cells * grid_enum::momentum_y];
  loaded_data.momentum.z = dev_conserved[cell_id + n_cells * grid_enum::momentum_z];
  loaded_data.energy     = dev_conserved[cell_id + n_cells * grid_enum::Energy];

#ifdef MHD
  // These are all cell centered values
  loaded_data.magnetic = mhd::utils::cellCenteredMagneticFields(dev_conserved, cell_id, xid, yid, zid, n_cells, nx, ny);
#endif  // MHD

#ifdef DE
  loaded_data.gas_energy = dev_conserved[cell_id + n_cells * grid_enum::GasEnergy];
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    loaded_data.scalar[i] = dev_conserved[cell_id + n_cells * (grid_enum::scalar + i)];
  }
#endif  // SCALAR

  // Now that all the data is loaded, let's sort out the direction
  // if constexpr(dir == 0) in this case everything is already set so we'll skip this case
  if constexpr (dir == 1) {
    math_utils::Cyclic_Permute_Once(loaded_data.momentum);
#ifdef MHD
    math_utils::Cyclic_Permute_Once(loaded_data.magnetic);
#endif  // MHD
  } else if constexpr (dir == 2) {
    math_utils::Cyclic_Permute_Twice(loaded_data.momentum);
#ifdef MHD
    math_utils::Cyclic_Permute_Twice(loaded_data.magnetic);
#endif  // MHD
  }

  return loaded_data;
}
// =====================================================================================================================

// =====================================================================================================================
__inline__ __host__ __device__ Primitive Conserved_2_Primitive(Conserved const &conserved_in, Real const gamma)
{
  Primitive output;

  // First the easy ones
  output.density    = conserved_in.density;
  output.velocity.x = conserved_in.momentum.x / conserved_in.density;
  output.velocity.y = conserved_in.momentum.y / conserved_in.density;
  output.velocity.z = conserved_in.momentum.z / conserved_in.density;

#ifdef MHD
  output.magnetic.x = conserved_in.magnetic.x;
  output.magnetic.y = conserved_in.magnetic.y;
  output.magnetic.z = conserved_in.magnetic.z;
#endif  // MHD

#ifdef DE
  output.gas_energy_specific = conserved_in.gas_energy / conserved_in.density;
#endif  // DE

#ifdef SCALAR
  for (size_t i = 0; i < grid_enum::nscalars; i++) {
    output.scalar_specific[i] = conserved_in.scalar[i] / conserved_in.density;
  }
#endif  // SCALAR

// Now that the easy ones are done let's figure out the pressure
#ifdef DE  // DE
  Real E_non_thermal = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(output.density, output.velocity.x,
                                                                          output.velocity.y, output.velocity.z);

  #ifdef MHD
  E_non_thermal += mhd::utils::computeMagneticEnergy(output.magnetic.x, output.magnetic.y, output.magnetic.z);
  #endif  // MHD

  output.pressure = hydro_utilities::Get_Pressure_From_DE(conserved_in.energy, conserved_in.energy - E_non_thermal,
                                                          conserved_in.gas_energy, gamma);
#else  // not DE
  #ifdef MHD
  output.pressure = hydro_utilities::Calc_Pressure_Primitive(
      conserved_in.energy, conserved_in.density, output.velocity.x, output.velocity.y, output.velocity.z, gamma,
      output.magnetic.x, output.magnetic.y, output.magnetic.z);
  #else   // not MHD
  output.pressure = hydro_utilities::Calc_Pressure_Primitive(
      conserved_in.energy, conserved_in.density, output.velocity.x, output.velocity.y, output.velocity.z, gamma);
  #endif  // MHD
#endif    // DE

  return output;
}
// =====================================================================================================================

// =====================================================================================================================
// Primitive_2_Conserved
// =====================================================================================================================

// =====================================================================================================================
// Load_Cell_Primitive
// =====================================================================================================================

}  // namespace hydro_utilities
