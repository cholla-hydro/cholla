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
  Real gas_energy_specific
#endif  // DE

#ifdef SCALAR
      Real scalar_specific[grid_enum::nscalars];
#endif  // SCALAR
};
}  // namespace reconstruction

#endif  //! RECONSTRUCTION_H
