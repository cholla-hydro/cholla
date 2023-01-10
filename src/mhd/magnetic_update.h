/*!
 * \file magnetic_update.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration of the kernel to update the magnetic field.
 * Method from Stone & Gardiner 2009 "A simple unsplit Godunov method for
 * multidimensional MHD" hereafter referred to as "S&G 2009"
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
 * \brief Namespace for MHD code
 *
 */
namespace mhd
{
    // =========================================================================
    /*!
     * \brief Update the magnetic field using the CT electric fields
     *
     * \param[in] sourceGrid The array which holds the old values of the
     * magnetic field
     * \param[out] destinationGrid The array to hold the updated values of the
     * magnetic field
     * \param[in] ctElectricFields The array of constrained transport electric
     * fields
     * \param[in] nx The number of cells in the x-direction
     * \param[in] ny The number of cells in the y-direction
     * \param[in] nz The number of cells in the z-direction
     * \param[in] n_cells The total number of cells
     * \param[in] dt The time step. If doing the half time step update make sure
     * to divide it by two when passing the time step to this kernel
     * \param[in] dx The size of each cell in the x-direction
     * \param[in] dy The size of each cell in the y-direction
     * \param[in] dz The size of each cell in the z-direction
     */
    __global__ void Update_Magnetic_Field_3D(Real *sourceGrid,
                                             Real *destinationGrid,
                                             Real *ctElectricFields,
                                             int const nx,
                                             int const ny,
                                             int const nz,
                                             int const n_cells,
                                             Real const dt,
                                             Real const dx,
                                             Real const dy,
                                             Real const dz);
    // =========================================================================
} // end namespace mhd