/*!
 * \file mhd_utilities.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of various utility functions for MHD
 *
 */

#pragma once

// STL Includes
#include <vector>

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "../utils/basic_structs.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/math_utilities.h"

namespace mhd::utils
{
/*!
 * \brief Namespace for functions required by functions within the mhd::utils
 * namespace. Everything in this name space should be regarded as private
 * but is made accesible for testing
 *
 */
namespace internal
{
// =====================================================================
/*!
 * \brief Compute the fast or slow magnetosonic wave speeds
 *
 * \param density The density
 * \param gasPressure The gas pressure
 * \param magneticX Magnetic field in the x-direction
 * \param magneticY Magnetic field in the y-direction
 * \param magneticZ Magnetic field in the z-direction
 * \param gamma The adiabatic index
 * \param waveChoice Which speed to compute. If +1 then compute the
 * speed of the fast magnetosonic wave, if -1 then the speed of the slow
 * magnetosonic wave
 * \return Real The speed of the fast or slow magnetosonic wave
 */
inline __host__ __device__ Real _magnetosonicSpeed(Real const &density, Real const &gasPressure, Real const &magneticX,
                                                   Real const &magneticY, Real const &magneticZ, Real const &gamma,
                                                   Real const &waveChoice)
{
  // Compute the sound speed
  Real bXSquared = magneticX * magneticX;
  Real bSquared  = bXSquared + ((magneticY * magneticY) + (magneticZ * magneticZ));

  Real term1 = gamma * gasPressure + bSquared;

  Real term2 = (term1 * term1) - 4. * gamma * gasPressure * bXSquared;
  term2      = sqrt(term2);

  return sqrt((term1 + waveChoice * term2) / (2.0 * fmax(density, TINY_NUMBER)));
}
// =====================================================================
}  // namespace internal

// =========================================================================
/*!
 * \brief Compute the magnetic energy
 *
 * \param[in] magneticX The magnetic field in the X-direction
 * \param[in] magneticY The magnetic field in the Y-direction
 * \param[in] magneticZ The magnetic field in the Z-direction
 * \return Real The magnetic energy
 */
inline __host__ __device__ Real computeMagneticEnergy(Real const &magneticX, Real const &magneticY,
                                                      Real const &magneticZ)
{
  return 0.5 * math_utils::SquareMagnitude(magneticX, magneticY, magneticZ);
}
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the MHD thermal energy in a cell
 *
 * \param[in] energyTot The total energy
 * \param[in] density The density
 * \param[in] momentumX Momentum in the x-direction
 * \param[in] momentumY Momentum in the y-direction
 * \param[in] momentumZ Momentum in the z-direction
 * \param[in] magneticX Magnetic field in the x-direction
 * \param[in] magneticY Magnetic field in the y-direction
 * \param[in] magneticZ Magnetic field in the z-direction
 * \param[in] gamma The adiabatic index
 * \return Real The thermal energy in a cell
 */
inline __host__ __device__ Real computeThermalEnergy(Real const &energyTot, Real const &density, Real const &momentumX,
                                                     Real const &momentumY, Real const &momentumZ,
                                                     Real const &magneticX, Real const &magneticY,
                                                     Real const &magneticZ, Real const &gamma)
{
  return energyTot - 0.5 * math_utils::SquareMagnitude(momentumX, momentumY, momentumZ) / fmax(density, TINY_NUMBER) -
         computeMagneticEnergy(magneticX, magneticY, magneticZ);
}
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the total MHD pressure. I.e. magnetic pressure + gas
 * pressure
 *
 * \param[in] gasPressure The gas pressure
 * \param[in] magneticX Magnetic field in the x-direction
 * \param[in] magneticY Magnetic field in the y-direction
 * \param[in] magneticZ Magnetic field in the z-direction
 * \return Real The total MHD pressure
 */
inline __host__ __device__ Real computeTotalPressure(Real const &gasPressure, Real const &magneticX,
                                                     Real const &magneticY, Real const &magneticZ)
{
  Real pTot = gasPressure + computeMagneticEnergy(magneticX, magneticY, magneticZ);

  return fmax(pTot, TINY_NUMBER);
}
/// Overload for Vector objects
inline __host__ __device__ Real computeTotalPressure(Real const &gasPressure, hydro_utilities::Vector const &magnetic)
{
  return computeTotalPressure(gasPressure, magnetic.x, magnetic.y, magnetic.z);
}
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the speed of the fast magnetosonic wave
 *
 * \param density The gas pressure
 * \param pressure The density
 * \param magneticX Magnetic field in the x-direction
 * \param magneticY Magnetic field in the y-direction
 * \param magneticZ Magnetic field in the z-direction
 * \param gamma The adiabatic index
 * \return Real The speed of the fast magnetosonic wave
 */
inline __host__ __device__ Real fastMagnetosonicSpeed(Real const &density, Real const &pressure, Real const &magneticX,
                                                      Real const &magneticY, Real const &magneticZ, Real const &gamma)
{
  // Compute the sound speed
  return mhd::utils::internal::_magnetosonicSpeed(density, pressure, magneticX, magneticY, magneticZ, gamma, 1.0);
}
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the speed of the slow magnetosonic wave
 *
 * \param density The gas pressure
 * \param pressure The density
 * \param magneticX Magnetic field in the x-direction
 * \param magneticY Magnetic field in the y-direction
 * \param magneticZ Magnetic field in the z-direction
 * \param gamma The adiabatic index
 * \return Real The speed of the slow magnetosonic wave
 */
inline __host__ __device__ Real slowMagnetosonicSpeed(Real const &density, Real const &pressure, Real const &magneticX,
                                                      Real const &magneticY, Real const &magneticZ, Real const &gamma)
{
  // Compute the sound speed
  return mhd::utils::internal::_magnetosonicSpeed(density, pressure, magneticX, magneticY, magneticZ, gamma, -1.0);
}
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the speed of the Alfven wave in a cell
 *
 * \param[in] magneticX The magnetic field in the x direction, ie the direction
 * along with the Riemann solver is acting
 * \param[in] density The density in the cell
 * \return Real The Alfven wave speed
 */
inline __host__ __device__ Real alfvenSpeed(Real const &magneticX, Real const &density)
{
  // Compute the Alfven wave speed
  return fabs(magneticX) / sqrt(fmax(density, TINY_NUMBER));
}
// =========================================================================

// =========================================================================
#ifdef MHD
/*!
 * \brief Compute the cell centered average of the magnetic fields in a
 * given cell
 *
 * \param[in] dev_conserved A pointer to the device array of conserved variables
 * \param[in] id The 1D index into each grid subarray.
 * \param[in] xid The x index
 * \param[in] yid The y index
 * \param[in] zid The z index
 * \param[in] n_cells The total number of cells
 * \param[in] nx The number of cells in the x-direction
 * \param[in] ny The number of cells in the y-direction
 * \param[out] avgBx The cell centered average magnetic field in the x-direction
 * \param[out] avgBy The cell centered average magnetic field in the y-direction
 * \param[out] avgBz The cell centered average magnetic field in the z-direction
 *
 * \return hydro_utilities::Vector with the X, Y, and Z cell centered magnetic
 * fields. Can be called with structured binding like `auto [x, y,
 * z] = mhd::utils::cellCenteredMagneticFields(*args*)
 */
inline __host__ __device__ hydro_utilities::Vector cellCenteredMagneticFields(Real const *dev_conserved,
                                                                              size_t const &id, size_t const &xid,
                                                                              size_t const &yid, size_t const &zid,
                                                                              size_t const &n_cells, size_t const &nx,
                                                                              size_t const &ny)
{
  // Ternary operator to check that no values outside of the magnetic field
  // arrays are loaded. If the cell is on the edge that doesn't have magnetic
  // fields on both sides then instead set the centered magnetic field to be
  // equal to the magnetic field of the closest edge.
  Real avgBx = (xid > 0) ?
                         /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_x)*n_cells + id] +
                                            dev_conserved[(grid_enum::magnetic_x)*n_cells +
                                                          cuda_utilities::compute1DIndex(xid - 1, yid, zid, nx, ny)])
                         :
                         /*if false*/ dev_conserved[(grid_enum::magnetic_x)*n_cells + id];
  Real avgBy = (yid > 0) ?
                         /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_y)*n_cells + id] +
                                            dev_conserved[(grid_enum::magnetic_y)*n_cells +
                                                          cuda_utilities::compute1DIndex(xid, yid - 1, zid, nx, ny)])
                         :
                         /*if false*/ dev_conserved[(grid_enum::magnetic_y)*n_cells + id];
  Real avgBz = (zid > 0) ?
                         /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_z)*n_cells + id] +
                                            dev_conserved[(grid_enum::magnetic_z)*n_cells +
                                                          cuda_utilities::compute1DIndex(xid, yid, zid - 1, nx, ny)])
                         :
                         /*if false*/ dev_conserved[(grid_enum::magnetic_z)*n_cells + id];

  return {avgBx, avgBy, avgBz};
}
// =========================================================================

// =========================================================================
/*!
 * \brief Initialize the magnitice field from the vector potential
 *
 * \param H The Header struct
 * \param C The Conserved struct
 * \param vectorPotential The vector potential in the same format as the other arrays in Cholla
 */
void Init_Magnetic_Field_With_Vector_Potential(Header const &H, Grid3D::Conserved const &C,
                                               std::vector<Real> const &vectorPotential);
// =========================================================================
#endif  // MHD
}  // end namespace mhd::utils
