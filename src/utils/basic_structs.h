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
};
// =====================================================================================================================

// =====================================================================================================================
/*!
 * \brief A data only struct for the primitive variables
 *
 */
struct Primitive {
  // Hydro variables
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
            Vector const &in_magnetic = {0, 0, 0})
      : density(in_density), velocity(in_velocity), pressure(in_pressure)
  {
#ifdef MHD
    magnetic = in_magnetic;
#endif  // mhd
  };
};
// =====================================================================================================================
}  // namespace hydro_utilities