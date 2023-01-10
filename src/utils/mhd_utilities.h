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
#include "../utils/cuda_utilities.h"

namespace mhd{
namespace utils{
    /*!
     * \brief Namespace for functions required by functions within the mhd::utils
     * namespace. Everything in this name space should be regarded as private
     * but is made accesible for testing
     *
     */
    namespace _internal
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
    }// mhd::utils::_internal namespace

    // =========================================================================
    /*!
     * \brief Compute the energy in a cell. If MHD is not defined then simply
     * return the hydro only energy
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
        Real energy =  (fmax(pressure,TINY_NUMBER)/(gamma - 1.))
                       + 0.5 * density * (velocityX*velocityX + ((velocityY*velocityY) + (velocityZ*velocityZ)));
        #ifdef  MHD
            energy += 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ)));
        #endif  //MHD

        return energy;
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
     * \brief Compute the magnetic energy
     *
     * \param[in] magneticX The magnetic field in the X-direction
     * \param[in] magneticY The magnetic field in the Y-direction
     * \param[in] magneticZ The magnetic field in the Z-direction
     * \return Real The magnetic energy
     */
    inline __host__ __device__ Real computeMagneticEnergy(Real const &magneticX,
                                                          Real const &magneticY,
                                                          Real const &magneticZ)
    {
        return 0.5 * (magneticX*magneticX + ((magneticY*magneticY) + (magneticZ*magneticZ)));
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
        return mhd::utils::_internal::_magnetosonicSpeed(density,
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
        return mhd::utils::_internal::_magnetosonicSpeed(density,
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
     * \return Real local struct with the X, Y, and Z cell centered magnetic
     * fields. Intended to be called with structured binding like `auto [x, y,
     * z] = mhd::utils::cellCenteredMagneticFields(*args*)
     */
    inline __host__ __device__ auto cellCenteredMagneticFields(Real const *dev_conserved,
                                                               size_t const &id,
                                                               size_t const &xid,
                                                               size_t const &yid,
                                                               size_t const &zid,
                                                               size_t const &n_cells,
                                                               size_t const &nx,
                                                               size_t const &ny)
    {
        // Ternary operator to check that no values outside of the magnetic field
        // arrays are loaded. If the cell is on the edge that doesn't have magnetic
        // fields on both sides then instead set the centered magnetic field to be
        // equal to the magnetic field of the closest edge. T
        Real avgBx = (xid > 0) ?
            /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_x)*n_cells + id] + dev_conserved[(grid_enum::magnetic_x)*n_cells + cuda_utilities::compute1DIndex(xid-1, yid,   zid,   nx, ny)]):
            /*if false*/       dev_conserved[(grid_enum::magnetic_x)*n_cells + id];
        Real avgBy = (yid > 0) ?
            /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_y)*n_cells + id] + dev_conserved[(grid_enum::magnetic_y)*n_cells + cuda_utilities::compute1DIndex(xid,   yid-1, zid,   nx, ny)]):
            /*if false*/       dev_conserved[(grid_enum::magnetic_y)*n_cells + id];
        Real avgBz = (zid > 0) ?
            /*if true*/ 0.5 * (dev_conserved[(grid_enum::magnetic_z)*n_cells + id] + dev_conserved[(grid_enum::magnetic_z)*n_cells + cuda_utilities::compute1DIndex(xid,   yid,   zid-1, nx, ny)]):
            /*if false*/       dev_conserved[(grid_enum::magnetic_z)*n_cells + id];

        struct returnStruct
        {
            Real x, y, z;
        };
        return returnStruct{avgBx, avgBy, avgBz};
    }
    #endif // MHD
    // =========================================================================
} // end namespace mhd::utils
} // end namespace mhd