/*!
 * \file mhd_utilities.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of various utility functions for MHD and
 * for the various kernels, functions, and tools required for the 3D VL+CT MHD
 * integrator. Due to the CUDA/HIP compiler requiring that device functions be
 * directly accessible to the file they're used in most device functions will be
 * implemented in the header file
 *
 */

// STL Includes
#include <cfloat>
#include <limits>

// External Includes

// Local Includes
#include "../mhd/magnetic_divergence.h"
#include "../utils/cuda_utilities.h"
#include "../utils/reduction_utilities.h"
#include "../utils/DeviceVector.h"

namespace mhd
{
    // =========================================================================
    __global__ void calculateMagneticDivergence(Real const *dev_conserved,
                                                Real *dev_maxDivergence,
                                                Real const dx,
                                                Real const dy,
                                                Real const dz,
                                                int const nx,
                                                int const ny,
                                                int const nz,
                                                int const n_cells)
    {
        // Variables to store the divergence
        Real cellDivergence;
        Real maxDivergence = 0.0;

        // Index variables
        int xid, yid, zid, id_xMin1, id_yMin1, id_zMin1;

        // Grid stride loop to perform as much of the reduction as possible
        for(size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x)
        {
            // compute the real indices
            cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

            // Thread guard to avoid overrun and to skip ghost cells that cannot
            // have their divergences computed due to a missing face;
            if (    xid > 1  and yid > 1  and zid > 1
                and xid < nx and yid < ny and zid < nz)
            {
                // Compute the various offset indices
                id_xMin1 = cuda_utilities::compute1DIndex(xid-1, yid  , zid  , nx, ny);
                id_yMin1 = cuda_utilities::compute1DIndex(xid  , yid-1, zid  , nx, ny);
                id_zMin1 = cuda_utilities::compute1DIndex(xid  , yid  , zid-1, nx, ny);

                // Compute divergence
                cellDivergence =
                    ((   dev_conserved[id       + (grid_enum::magnetic_x)*n_cells]
                       - dev_conserved[id_xMin1 + (grid_enum::magnetic_x)*n_cells])
                    / dx)
                    + (( dev_conserved[id       + (grid_enum::magnetic_y)*n_cells]
                       - dev_conserved[id_yMin1 + (grid_enum::magnetic_y)*n_cells])
                    / dy)
                    + (( dev_conserved[id       + (grid_enum::magnetic_z)*n_cells]
                       - dev_conserved[id_zMin1 + (grid_enum::magnetic_z)*n_cells])
                    / dz);

                maxDivergence = max(maxDivergence, fabs(cellDivergence));
            }
        }

        // Perform reduction across the entire grid
        reduction_utilities::gridReduceMax(maxDivergence, dev_maxDivergence);
    }
    // =========================================================================

    // =========================================================================
    Real launchCalculateMagneticDivergence(Real const *dev_conserved,
                                           Real const dx,
                                           Real const dy,
                                           Real const dz,
                                           int const nx,
                                           int const ny,
                                           int const nz,
                                           int const n_cells)
    {
        // First let's create some variables we'll need.
        cuda_utilities::AutomaticLaunchParams static const launchParams(mhd::calculateMagneticDivergence);
        cuda_utilities::DeviceVector<Real> static dev_maxDivergence(1);

        // Set the device side inverse time step to the smallest possible double
        // so that the reduction isn't using the maximum value of the previous
        // iteration
        dev_maxDivergence.assign(std::numeric_limits<Real>::lowest());

        // Now lets get the local maximum divergence
        hipLaunchKernelGGL(mhd::calculateMagneticDivergence,
                           launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0,
                           dev_conserved, dev_maxDivergence.data(),
                           dx, dy, dz,
                           nx, ny, nz,
                           n_cells);
        CudaCheckError();

        return dev_maxDivergence[0];
    }
    // =========================================================================
} // end namespace mhd