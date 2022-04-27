/*!
 * \file mhd_utilities.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of various utility functions for MHD
 *
 */

#pragma once

// STL Includes

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"

/*!
 * \brief Namespace for MHD utilities
 *
 */
namespace mhdUtils
{
    namespace // Anonymouse namespace
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
        inline __host__ __device__ Real _magnetosonicSpeed(Real const &density,
                                                           Real const &gasPressure,
                                                           Real const &magneticX,
                                                           Real const &magneticY,
                                                           Real const &magneticZ,
                                                           Real const &gamma,
                                                           Real const &waveChoice)
        {
            // Compute the sound speed
            Real bXSquared = magneticX * magneticX;
            Real bSquared  = bXSquared + ((magneticY*magneticY) + (magneticZ*magneticZ));

            Real term1 = gamma * gasPressure + bSquared;

            Real term2 = (term1*term1) - 4. * gamma * gasPressure * bXSquared;
            term2      = sqrt(term2);

            return sqrt( (term1 + waveChoice * term2) / (2.0 * fmax(density, TINY_NUMBER)) );
        }
        // =====================================================================
    }// Anonymouse namespace

    // =========================================================================
    /*!
     * \brief Compute the MHD energy in the cell
     *
     * \param[in] pressure The gas pressure
     * \param[in] density The density
     * \param[in] velocityX Velocity in the x-direction
     * \param[in] velocityY Velocity in the y-direction
     * \param[in] velocityZ Velocity in the z-direction
     * \param[in] magneticX Magnetic field in the x-direction
     * \param[in] magneticY Magnetic field in the y-direction
     * \param[in] magneticZ Magnetic field in the z-direction
     * \param[in] gamma The adiabatic index
     * \return Real The energy within a cell
     */
    inline __host__ __device__ Real computeEnergy(Real const &pressure,
                                                  Real const &density,
                                                  Real const &velocityX,
                                                  Real const &velocityY,
                                                  Real const &velocityZ,
                                                  Real const &magneticX,
                                                  Real const &magneticY,
                                                  Real const &magneticZ,
                                                  Real const &gamma)
    {
        // Compute and return energy
        return (fmax(pressure,TINY_NUMBER)/(gamma - 1.))
                + 0.5 * density * (velocityX*velocityX + ((velocityY*velocityY) + (velocityZ*velocityZ)))
                + 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ)));
    }
    // =========================================================================

    // =========================================================================
    /*!
     * \brief Compute the MHD gas pressure in a cell
     *
     * \param[in] energy The energy
     * \param[in] density The density
     * \param[in] momentumX Momentum in the x-direction
     * \param[in] momentumY Momentum in the y-direction
     * \param[in] momentumZ Momentum in the z-direction
     * \param[in] magneticX Magnetic field in the x-direction
     * \param[in] magneticY Magnetic field in the y-direction
     * \param[in] magneticZ Magnetic field in the z-direction
     * \param[in] gamma The adiabatic index
     * \return Real The gas pressure in a cell
     */
    inline __host__ __device__ Real computeGasPressure(Real const &energy,
                                                       Real const &density,
                                                       Real const &momentumX,
                                                       Real const &momentumY,
                                                       Real const &momentumZ,
                                                       Real const &magneticX,
                                                       Real const &magneticY,
                                                       Real const &magneticZ,
                                                       Real const &gamma)
    {
        Real pressure = (gamma - 1.)
                            * (energy
                                - 0.5 * (momentumX*momentumX + ((momentumY*momentumY) + (momentumZ*momentumZ))) / density
                                - 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ))));

        return fmax(pressure, TINY_NUMBER);
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
    inline __host__ __device__ Real computeThermalEnergy(Real const &energyTot,
                                                         Real const &density,
                                                         Real const &momentumX,
                                                         Real const &momentumY,
                                                         Real const &momentumZ,
                                                         Real const &magneticX,
                                                         Real const &magneticY,
                                                         Real const &magneticZ,
                                                         Real const &gamma)
    {
        return energyTot - 0.5 * (momentumX*momentumX + ((momentumY*momentumY) + (momentumZ*momentumZ))) / fmax(density,TINY_NUMBER)
                         - 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ)));
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
    inline __host__ __device__ Real computeTotalPressure(Real const &gasPressure,
                                                         Real const &magneticX,
                                                         Real const &magneticY,
                                                         Real const &magneticZ)
    {
        Real pTot =  gasPressure + 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ)));

        return fmax(pTot, TINY_NUMBER);
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
    inline __host__ __device__ Real fastMagnetosonicSpeed(Real const &density,
                                                          Real const &pressure,
                                                          Real const &magneticX,
                                                          Real const &magneticY,
                                                          Real const &magneticZ,
                                                          Real const &gamma)
    {
        // Compute the sound speed
        return _magnetosonicSpeed(density,
                                  pressure,
                                  magneticX,
                                  magneticY,
                                  magneticZ,
                                  gamma,
                                  1.0);
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
    inline __host__ __device__ Real slowMagnetosonicSpeed(Real const &density,
                                                          Real const &pressure,
                                                          Real const &magneticX,
                                                          Real const &magneticY,
                                                          Real const &magneticZ,
                                                          Real const &gamma)
    {
        // Compute the sound speed
        return _magnetosonicSpeed(density,
                                  pressure,
                                  magneticX,
                                  magneticY,
                                  magneticZ,
                                  gamma,
                                  -1.0);
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
    inline __host__ __device__ Real alfvenSpeed(Real const &magneticX,
                                                Real const &density)
    {
        // Compute the Alfven wave speed
        return fabs(magneticX) / sqrt(fmax(density,TINY_NUMBER));
    }
    // =========================================================================

    // =========================================================================
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
     */
    inline __host__ __device__ void cellCenteredMagneticFields(Real const *dev_conserved,
                                                               size_t const &id,
                                                               size_t const &xid,
                                                               size_t const &yid,
                                                               size_t const &zid,
                                                               size_t const &n_cells,
                                                               size_t const &nx,
                                                               size_t const &ny,
                                                               Real &avgBx,
                                                               Real &avgBy,
                                                               Real &avgBz)
    {
        avgBx = 0.5 * (dev_conserved[(5+NSCALARS)*n_cells + id] + dev_conserved[(5+NSCALARS)*n_cells + ((xid-1) + yid*nx     + zid*nx*ny)]);
        avgBy = 0.5 * (dev_conserved[(6+NSCALARS)*n_cells + id] + dev_conserved[(6+NSCALARS)*n_cells + (xid     + (yid-1)*nx + zid*nx*ny)]);
        avgBz = 0.5 * (dev_conserved[(7+NSCALARS)*n_cells + id] + dev_conserved[(7+NSCALARS)*n_cells + (xid     + yid*nx     + (zid-1)*nx*ny)]);
    }
    // =========================================================================

} // end  namespace mhdUtils