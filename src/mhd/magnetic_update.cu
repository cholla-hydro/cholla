/*!
 * \file magnetic_update.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the definition of the kernel to update the magnetic field
 *
 */

// STL Includes

// External Includes

// Local Includes
#include "../mhd/magnetic_update.h"
#include "../utils/cuda_utilities.h"

namespace mhd
{
    // =========================================================================
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
                                             Real const dz)
    {
        // get a thread index
        int const blockId  = blockIdx.x + blockIdx.y*gridDim.x;
        int const threadId = threadIdx.x + blockId * blockDim.x;
        int xid, yid, zid;
        cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

        // Thread guard to avoid overrun and to skip ghost cells that cannot be
        // evolved due to missing electric fields that can't be reconstructed
        if (    xid < nx-1
            and yid < ny-1
            and zid < nz-1)
        {
            // Compute the three dt/dx quantities
            Real const dtodx = dt/dx;
            Real const dtody = dt/dy;
            Real const dtodz = dt/dz;

            // Load the various edge electric fields required. The '1' and '2'
            // fields are not shared and the '3' fields are shared by two of the
            // updates
            Real electric_x_1 = ctElectricFields[(cuda_utilities::compute1DIndex(xid  , yid+1, zid  , nx, ny))];
            Real electric_x_2 = ctElectricFields[(cuda_utilities::compute1DIndex(xid  , yid  , zid+1, nx, ny))];
            Real electric_x_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid  , yid+1, zid+1, nx, ny))];
            Real electric_y_1 = ctElectricFields[(cuda_utilities::compute1DIndex(xid+1, yid  , zid  , nx, ny)) + n_cells];
            Real electric_y_2 = ctElectricFields[(cuda_utilities::compute1DIndex(xid  , yid  , zid+1, nx, ny)) + n_cells];
            Real electric_y_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid+1, yid  , zid+1, nx, ny)) + n_cells];
            Real electric_z_1 = ctElectricFields[(cuda_utilities::compute1DIndex(xid+1, yid  , zid  , nx, ny)) + 2 * n_cells];
            Real electric_z_2 = ctElectricFields[(cuda_utilities::compute1DIndex(xid  , yid+1, zid  , nx, ny)) + 2 * n_cells];
            Real electric_z_3 = ctElectricFields[(cuda_utilities::compute1DIndex(xid+1, yid+1, zid  , nx, ny)) + 2 * n_cells];

            // Perform Updates

            // X field update
            destinationGrid[threadId + (grid_enum::magnetic_x)*n_cells] = sourceGrid[threadId + (grid_enum::magnetic_x)*n_cells]
                + dtodz * (electric_y_3 - electric_y_1)
                + dtody * (electric_z_1 - electric_z_3);

            // Y field update
            destinationGrid[threadId + (grid_enum::magnetic_y)*n_cells] = sourceGrid[threadId + (grid_enum::magnetic_y)*n_cells]
                + dtodx * (electric_z_3 - electric_z_2)
                + dtodz * (electric_x_1 - electric_x_3);

            // Z field update
            destinationGrid[threadId + (grid_enum::magnetic_z)*n_cells] = sourceGrid[threadId + (grid_enum::magnetic_z)*n_cells]
                + dtody * (electric_x_3 - electric_x_2)
                + dtodx * (electric_y_2 - electric_y_3);
        }
    }
    // =========================================================================
} // end namespace mhd