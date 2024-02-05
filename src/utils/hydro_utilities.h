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
  Vector velocity;

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
 * \brief A data only struct for the primtive variables
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
  Real scalar[grid_enum::nscalars];
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

}  // namespace hydro_utilities
