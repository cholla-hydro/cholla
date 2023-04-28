/*!
 * \file magnetic_update_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the magnetic update code
 *
 */

// STL Includes
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../mhd/magnetic_update.h"
#include "../utils/cuda_utilities.h"
#include "../utils/testing_utilities.h"

#ifdef MHD
// =============================================================================
/*!
 * \brief Test fixture for tMHDUpdateMagneticField3D test suite
 *
 */
class tMHDUpdateMagneticField3D : public ::testing::Test
{
 public:
  /*!
   * \brief Initialize and allocate all the various required variables and
   * arrays
   *
   */
  tMHDUpdateMagneticField3D()
      : nx(3),
        ny(nx),
        nz(nx),
        n_cells(nx * ny * nz),
        dt(3.2),
        dx(2.5),
        dy(2.5),
        dz(2.5),
        sourceGrid(n_cells * (grid_enum::num_fields)),
        destinationGrid(n_cells * (grid_enum::num_fields), -999.),
        ctElectricFields(n_cells * 3),
        fiducialData(n_cells * (grid_enum::num_fields), -999.),
        dimGrid((n_cells + TPB - 1) / TPB, 1, 1),
        dimBlock(TPB, 1, 1)
  {
    // Allocate device arrays
    CudaSafeCall(cudaMalloc(&dev_sourceGrid, sourceGrid.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_destinationGrid, destinationGrid.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_ctElectricFields, ctElectricFields.size() * sizeof(double)));

    // Populate the grids with values where vector.at(i) = double(i). The
    // values chosen aren't that important, just that every cell has a unique
    // value
    std::iota(std::begin(sourceGrid), std::end(sourceGrid), 0.);
    std::iota(std::begin(ctElectricFields), std::end(ctElectricFields), sourceGrid.back() + 1);
  }
  ~tMHDUpdateMagneticField3D() = default;

 protected:
  // Initialize the test grid and other state variables
  size_t const nx, ny, nz;
  size_t const n_cells;
  Real const dt, dx, dy, dz;

  // Launch Parameters
  dim3 const dimGrid;   // How many blocks in the grid
  dim3 const dimBlock;  // How many threads per block

  // Make sure the vector is large enough that the locations where the
  // magnetic field would be in the real grid are filled
  std::vector<double> sourceGrid;
  std::vector<double> destinationGrid;
  std::vector<double> ctElectricFields;
  std::vector<double> fiducialData;

  // device pointers
  double *dev_sourceGrid, *dev_destinationGrid, *dev_ctElectricFields, *dev_fiducialData;

  /*!
   * \brief Launch the kernel and check results
   *
   */
  void runTest()
  {
    // Copy values to GPU
    CudaSafeCall(
        cudaMemcpy(dev_sourceGrid, sourceGrid.data(), sourceGrid.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_destinationGrid, destinationGrid.data(), destinationGrid.size() * sizeof(Real),
                            cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_ctElectricFields, ctElectricFields.data(), ctElectricFields.size() * sizeof(Real),
                            cudaMemcpyHostToDevice));

    // Call the kernel to test
    hipLaunchKernelGGL(mhd::Update_Magnetic_Field_3D, dimGrid, dimBlock, 0, 0, dev_sourceGrid, dev_destinationGrid,
                       dev_ctElectricFields, nx, ny, nz, n_cells, dt, dx, dy, dz);
    CudaCheckError();

    // Copy test data back
    CudaSafeCall(cudaMemcpy(destinationGrid.data(), dev_destinationGrid, destinationGrid.size() * sizeof(Real),
                            cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Check the results
    for (size_t i = 0; i < fiducialData.size(); i++) {
      int xid, yid, zid;
      cuda_utilities::compute3DIndices(i, nx, ny, xid, yid, zid);
      testingUtilities::checkResults(fiducialData.at(i), destinationGrid.at(i),
                                     "value at i = " + std::to_string(i) + ", xid  = " + std::to_string(xid) +
                                         ", yid  = " + std::to_string(yid) + ", zid  = " + std::to_string(zid));
    }
  }
};
// =============================================================================

// =============================================================================
TEST_F(tMHDUpdateMagneticField3D, CorrectInputExpectCorrectOutput)
{
  // Fiducial values
  fiducialData.at(148) = 155.68000000000001;
  fiducialData.at(175) = 164.75999999999999;
  fiducialData.at(202) = 204.56;

  // Launch kernel and check results
  runTest();
}
// =============================================================================
#endif  // MHD
