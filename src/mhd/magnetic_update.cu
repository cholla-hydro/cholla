/*!
 * \file magnetic_update.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the definition of the kernel to update the magnetic field.
 * Method from Stone & Gardiner 2009 "A simple unsplit Godunov method for
 * multidimensional MHD" hereafter referred to as "S&G 2009"
 *
 */

// STL Includes

// External Includes

// Local Includes
#include "../mhd/magnetic_update.h"
#include "../utils/cuda_utilities.h"
#ifdef MHD
namespace mhd
{
// =========================================================================
__global__ void Update_Magnetic_Field_3D(Real *sourceGrid, Real *destinationGrid, Real *ctElectricFields, int const nx,
                                         int const ny, int const nz, int const n_cells, Real const dt, Real const dx,
                                         Real const dy, Real const dz)
{
  // get a thread index
  int const blockId  = blockIdx.x + blockIdx.y * gridDim.x;
  int const threadId = threadIdx.x + blockId * blockDim.x;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

  // Thread guard to avoid overrun and to skip ghost cells that cannot be
  // evolved due to missing electric fields that can't be reconstructed
  if (xid < nx - 2 and yid < ny - 2 and zid < nz - 2) {
    // Compute the three dt/dx quantities
    Real const dtodx = dt / dx;
    Real const dtody = dt / dy;
    Real const dtodz = dt / dz;

    // Load the various edge electric fields required. The '1' and '2'
    // fields are not shared and the '3' fields are shared by two of the
    // updates
    Real electric_x_1 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid, yid + 1, zid, nx, ny)) +
                                         grid_enum::ct_elec_x * n_cells];
    Real electric_x_2 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid, yid, zid + 1, nx, ny)) +
                                         grid_enum::ct_elec_x * n_cells];
    Real electric_x_3 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid, yid + 1, zid + 1, nx, ny)) +
                                         grid_enum::ct_elec_x * n_cells];
    Real electric_y_1 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid + 1, yid, zid, nx, ny)) +
                                         grid_enum::ct_elec_y * n_cells];
    Real electric_y_2 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid, yid, zid + 1, nx, ny)) +
                                         grid_enum::ct_elec_y * n_cells];
    Real electric_y_3 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid + 1, yid, zid + 1, nx, ny)) +
                                         grid_enum::ct_elec_y * n_cells];
    Real electric_z_1 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid + 1, yid, zid, nx, ny)) +
                                         grid_enum::ct_elec_z * n_cells];
    Real electric_z_2 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid, yid + 1, zid, nx, ny)) +
                                         grid_enum::ct_elec_z * n_cells];
    Real electric_z_3 = ctElectricFields[(cuda_utilities::compute1DIndex(
                                             xid + 1, yid + 1, zid, nx, ny)) +
                                         grid_enum::ct_elec_z * n_cells];

    // Perform Updates

    // X field update
    // S&G 2009 equation 10
    destinationGrid[threadId + grid_enum::magnetic_x * n_cells] =
        sourceGrid[threadId + grid_enum::magnetic_x * n_cells] +
        dtodz * (electric_y_3 - electric_y_1) +
        dtody * (electric_z_1 - electric_z_3);

    // Y field update
    // S&G 2009 equation 11
    destinationGrid[threadId + grid_enum::magnetic_y * n_cells] =
        sourceGrid[threadId + grid_enum::magnetic_y * n_cells] +
        dtodx * (electric_z_3 - electric_z_2) +
        dtodz * (electric_x_1 - electric_x_3);

    // Z field update
    // S&G 2009 equation 12
    destinationGrid[threadId + grid_enum::magnetic_z * n_cells] =
        sourceGrid[threadId + grid_enum::magnetic_z * n_cells] +
        dtody * (electric_x_3 - electric_x_2) +
        dtodx * (electric_y_2 - electric_y_3);
  }
}
// =========================================================================
}  // end namespace mhd
#endif  // MHD
