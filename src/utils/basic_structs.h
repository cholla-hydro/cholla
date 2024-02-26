/*!
 * \file basic_structs.h
 * \brief Constains some basic structs to be used around the code. Mostly this is here instead of hydro_utilities.h to
 * avoid circulary dependencies with mhd_utils.h
 *
 */

#pragma once

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

namespace hydro_utilities
{
// =====================================================================================================================
// Here are some basic structs that can be used in various places when needed
// =====================================================================================================================
/*!
 * \brief A data only struct that contains the Real members x, y, and z for usage as a vector
 *
 */
struct Vector {
  Real x, y, z;
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A data only struct for the conserved variables
 *
 */
struct Conserved {
  // Hydro variables
  Real density, energy;
  Vector momentum;

#ifdef MHD
  // These are all cell centered values
  Vector magnetic;
#endif  // MHD

#ifdef DE
  Real gas_energy;
#endif  // DE

#ifdef SCALAR
  Real scalar[grid_enum::nscalars];
#endif  // SCALAR

  /// Default constructor, should init everything to zero
  Conserved() = default;
  /// Manual constructor, mostly used for testing and doesn't init all members
  Conserved(Real const in_density, Vector const &in_momentum, Real const in_energy,
            Vector const &in_magnetic = {0, 0, 0}, Real const in_gas_energy = 0.0)
      : density(in_density), momentum(in_momentum), energy(in_energy)
  {
#ifdef MHD
    magnetic = in_magnetic;
#endif  // mhd

#ifdef DE
    gas_energy = in_gas_energy;
#endif  // DE
  };
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A data only struct for the primitive variables
 *
 */
struct Primitive {
  // Hydro variable
  Real density, pressure;
  Vector velocity;

#ifdef MHD
  // These are all cell centered values
  Vector magnetic;
#endif  // MHD

#ifdef DE
  /// The specific thermal energy in the gas
  Real gas_energy_specific;
#endif  // DE

#ifdef SCALAR
  Real scalar_specific[grid_enum::nscalars];
#endif  // SCALAR

  /// Default constructor, should init everything to zero
  Primitive() = default;
  /// Manual constructor, mostly used for testing and doesn't init all members
  Primitive(Real const in_density, Vector const &in_velocity, Real const in_pressure,
            Vector const &in_magnetic = {0, 0, 0}, Real const in_gas_energy_specific = 0.0)
      : density(in_density), velocity(in_velocity), pressure(in_pressure)
  {
#ifdef MHD
    magnetic = in_magnetic;
#endif  // mhd

#ifdef DE
    gas_energy_specific = in_gas_energy_specific;
#endif  // DE
  };
};
// =====================================================================================================================
}  // namespace hydro_utilities

namespace reconstruction
{
struct InterfaceState {
  // Hydro variables
  Real density, energy;
  /// Note that `pressure` here is the gas pressure not the total pressure which would include the magnetic component
  Real pressure;
  hydro_utilities::Vector velocity, momentum;

#ifdef MHD
  // These are all cell centered values
  Real total_pressure;
  hydro_utilities::Vector magnetic;
#endif  // MHD

#ifdef DE
  Real gas_energy_specific;
#endif  // DE

#ifdef SCALAR
  Real scalar_specific[grid_enum::nscalars];
#endif  // SCALAR

  // Define the constructors
  /// Default constructor, should set everything to 0
  InterfaceState() = default;
  /// Initializing constructor: used to initialize to specific values, mostly used in tests. It only initializes a
  /// subset of the member variables since that is what is used in tests at the time of writing.
  InterfaceState(Real const in_density, hydro_utilities::Vector const in_velocity, Real const in_energy,
                 Real const in_pressure, hydro_utilities::Vector const in_magnetic = {0, 0, 0},
                 Real const in_total_pressure = 0.0)
      : density(in_density), velocity(in_velocity), energy(in_energy), pressure(in_pressure)
  {
    momentum.x = velocity.x * density;
    momentum.y = velocity.y * density;
    momentum.z = velocity.z * density;
#ifdef MHD
    magnetic       = in_magnetic;
    total_pressure = in_total_pressure;
#endif  // MHD
#ifdef DE
    gas_energy_specific = 0.0;
#endif  // DE
  };
};
}  // namespace reconstruction
