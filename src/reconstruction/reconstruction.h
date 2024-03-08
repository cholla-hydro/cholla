/*!
 * \file reconstruction.h
 * \brief Contains the "riemann solved facing" controller function to choose, compute, and return the interface states
 *
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "../utils/hydro_utilities.h"

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
  };
};
}  // namespace reconstruction

#endif  //! RECONSTRUCTION_H