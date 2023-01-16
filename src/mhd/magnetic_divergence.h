/*!
 * \file magnetic_divergence.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the declaration for the functions that compute the magnetic
 * divergence
 *
 */

#pragma once

// STL Includes

// External Includes

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../grid/grid3D.h"
#include "../utils/gpu.hpp"

/*!
 * \brief Namespace for MHD code
 *
 */
namespace mhd
{
// =========================================================================
/*!
 * \brief Kernel to compute the maximum divergence of the magnetic field in
 * the grid. Uses `reduction_utilities::gridReduceMax` and as such should be
 * called with the minimum number of blocks. Recommend using the occupancy
 * API
 *
 * \param[in] dev_conserved The device array of conserved variables
 * \param[out] maxDivergence The device scalar to store the reduced divergence
 * at \param[in] dx Cell size in the X-direction \param[in] dy Cell size in the
 * Y-direction \param[in] dz Cell size in the Z-direction \param[in] nx Number
 * of cells in the X-direction \param[in] ny Number of cells in the Y-direction
 * \param[in] nz Number of cells in the Z-direction
 * \param[in] n_cells Total number of cells
 */
__global__ void calculateMagneticDivergence(Real const *dev_conserved,
                                            Real *maxDivergence, Real const dx,
                                            Real const dy, Real const dz,
                                            int const nx, int const ny,
                                            int const nz, int const n_cells);
// =========================================================================

// =========================================================================
/*!
 * \brief Compute the maximum magnetic divergence in the grid and report
 * an error if it exceeds the magnetic divergence limit or is negative. The
 * magnetic divergence limit is 1E-14 as determined by Athena as a
 * reasonable upper bound for correctness.
 *
 * \param G The grid object
 * \return Real The maximum magnetic divergence found in the grid. Can
 * usually be ignored since all checking is done in the fucntion, mostly
 * this return is for testing.
 */
Real checkMagneticDivergence(Grid3D const &G);
// =========================================================================
}  // end namespace mhd