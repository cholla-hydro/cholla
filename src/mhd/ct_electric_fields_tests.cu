/*!
 * \file ct_electric_fields_tests.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the CT electric fields
 *
 */

// STL Includes
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../io/io.h"
#include "../mhd/ct_electric_fields.h"
#include "../utils/testing_utilities.h"

#ifdef MHD
// =============================================================================
// Tests for the mhd::Calculate_CT_Electric_Fields kernel
// =============================================================================

// =============================================================================
/*!
 * \brief Test fixture for tMHDCalculateCTElectricFields test suite
 *
 */
class tMHDCalculateCTElectricFields : public ::testing::Test
{
 public:
  /*!
   * \brief Initialize and allocate all the various required variables and
   * arrays
   *
   */
  tMHDCalculateCTElectricFields()
      : n_cells(nx * ny * nz),
        fluxX(n_cells * (grid_enum::num_flux_fields)),
        fluxY(n_cells * (grid_enum::num_flux_fields)),
        fluxZ(n_cells * (grid_enum::num_flux_fields)),
        grid(n_cells * (grid_enum::num_fields)),
        testCTElectricFields(n_cells * 3, -999.),
        fiducialData(n_cells * 3, -999.),
        dimGrid((n_cells + TPB - 1) / TPB, 1, 1),
        dimBlock(TPB, 1, 1)
  {
    // Allocate device arrays
    CudaSafeCall(cudaMalloc(&dev_fluxX, fluxX.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_fluxY, fluxY.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_fluxZ, fluxZ.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_grid, grid.size() * sizeof(double)));
    CudaSafeCall(cudaMalloc(&dev_testCTElectricFields, testCTElectricFields.size() * sizeof(double)));

    // Populate the grids with values where vector.at(i) = double(i). The
    // values chosen aren't that important, just that every cell has a unique
    // value
    std::iota(std::begin(fluxX), std::end(fluxX), 0.);
    std::iota(std::begin(fluxY), std::end(fluxY), fluxX.back() + 1);
    std::iota(std::begin(fluxZ), std::end(fluxZ), fluxY.back() + 1);
    std::iota(std::begin(grid), std::end(grid), fluxZ.back() + 1);
  }
  ~tMHDCalculateCTElectricFields() = default;

 protected:
  // Initialize the test grid and other state variables
  size_t const nx = 2, ny = nx, nz = nx;
  size_t const n_cells;

  // Launch Parameters
  dim3 const dimGrid;   // How many blocks in the grid
  dim3 const dimBlock;  // How many threads per block

  // Make sure the vector is large enough that the locations where the
  // magnetic field would be in the real grid are filled
  std::vector<double> fluxX;
  std::vector<double> fluxY;
  std::vector<double> fluxZ;
  std::vector<double> grid;
  std::vector<double> testCTElectricFields;
  std::vector<double> fiducialData;

  // device pointers
  double *dev_fluxX, *dev_fluxY, *dev_fluxZ, *dev_grid, *dev_testCTElectricFields;

  /*!
   * \brief Launch the kernel and check results
   *
   */
  void Run_Test()
  {
    // Copy values to GPU
    CudaSafeCall(cudaMemcpy(dev_fluxX, fluxX.data(), fluxX.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_fluxY, fluxY.data(), fluxY.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_fluxZ, fluxZ.data(), fluxZ.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_grid, grid.data(), grid.size() * sizeof(Real), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(dev_testCTElectricFields, testCTElectricFields.data(),
                            testCTElectricFields.size() * sizeof(Real), cudaMemcpyHostToDevice));

    // Call the kernel to test
    hipLaunchKernelGGL(mhd::Calculate_CT_Electric_Fields, dimGrid, dimBlock, 0, 0, dev_fluxX, dev_fluxY, dev_fluxZ,
                       dev_grid, dev_testCTElectricFields, nx, ny, nz, n_cells);
    CudaCheckError();

    // Copy test data back
    CudaSafeCall(cudaMemcpy(testCTElectricFields.data(), dev_testCTElectricFields,
                            testCTElectricFields.size() * sizeof(Real), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();

    // Check the results
    for (size_t i = 0; i < fiducialData.size(); i++) {
      int xid, yid, zid;
      cuda_utilities::compute3DIndices(i, nx, ny, xid, yid, zid);
      testingUtilities::Check_Results(fiducialData.at(i), testCTElectricFields.at(i),
                                      "value at i = " + std::to_string(i) + ", xid  = " + std::to_string(xid) +
                                          ", yid  = " + std::to_string(yid) + ", zid  = " + std::to_string(zid));
    }
  }
};
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields, PositiveVelocityExpectCorrectOutput)
{
  // Fiducial values
  fiducialData.at(7)  = 60.951467108788492;
  fiducialData.at(15) = -98.736587665919359;
  fiducialData.at(23) = 61.768055665002557;

  // Launch kernel and check results
  Run_Test();
}
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields, NegativeVelocityExpectCorrectOutput)
{
  // Fiducial values
  fiducialData.at(7)  = 59.978246483260179;
  fiducialData.at(15) = -97.279949010457187;
  fiducialData.at(23) = 61.280813140085613;

  // Set the density fluxes to be negative to indicate a negative velocity
  // across the face
  for (size_t i = 0; i < n_cells; i++) {
    fluxX.at(i) = -fluxX.at(i);
    fluxY.at(i) = -fluxY.at(i);
    fluxZ.at(i) = -fluxZ.at(i);
  }

  // Launch kernel and check results
  Run_Test();
}
// =============================================================================

// =============================================================================
TEST_F(tMHDCalculateCTElectricFields, ZeroVelocityExpectCorrectOutput)
{
  // Fiducial values
  fiducialData.at(7)  = 60.464856796024335;
  fiducialData.at(15) = -98.008268338188287;
  fiducialData.at(23) = 61.524434402544081;

  // Set the density fluxes to be negative to indicate a negative velocity
  // across the face
  for (size_t i = 0; i < n_cells; i++) {
    fluxX.at(i) = 0.0;
    fluxY.at(i) = 0.0;
    fluxZ.at(i) = 0.0;
  }

  // Launch kernel and check results
  Run_Test();
}
// =============================================================================

// =============================================================================
TEST(tMHDCTSlope, CorrectInputExpectCorrectOutput)
{
  // Set up the basic parameters
  size_t const nx      = 5;
  size_t const ny      = nx;
  size_t const nz      = nx;
  int const xid        = nx / 2;
  int const yid        = ny / 2;
  int const zid        = nz / 2;
  size_t const n_cells = nx * ny * nz;

  // Set up the grid
  std::vector<double> flux(grid_enum::num_fields * n_cells), conserved(grid_enum::num_fields * n_cells);

  std::mt19937 prng(1);
  std::uniform_real_distribution<double> doubleRand(-5, 5);

  for (double& conserved_data : conserved) {
    conserved_data = doubleRand(prng);
  }
  for (double& flux_data : flux) {
    flux_data = doubleRand(prng);
  }

  // Fiducial data
  std::vector<double> fiducial_data = {
      -6.8725060451062561, -77.056763568617669, 1.4564238051915397,  5.4541656143291437,  -0.83503550003671911,
      -78.091781647940934, -2.6187125848387525, -5.6934594000939542, -16.243259069749971, -59.321631150095314,
      0.99291378610068892, 4.4004574252725384,  -1.6902722376320516, -63.074645759822637, -4.5776373499662899,
      -19.476095152639683, -2.0173881091784471, -74.484407919605786, -7.8184484634991724, -0.23206265131850434,
      0.41622472388590037, -74.479121547383727, -6.9903417764222358, -1.832282425083853};

  // Get test data. Only test the options that will be used
  std::vector<double> test_data;
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 0, 2, -1, 1, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 0, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 0, 1, 2, 1, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 0, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 0, 1, -1, 1, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 0, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 0, 1, 2, 1, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 0, 2, -1, -1, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 1, 2, -1, 0, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 1, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 1, 0, 2, 0, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 1, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 1, 0, -1, 0, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 1, -1, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 1, 0, 2, 0, 2, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 1, 2, -1, 2, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 2, 0, -1, 0, 1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 2, -1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 2, 0, 1, 0, 1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), 1, 2, 1, -1, 1, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 2, 1, -1, 0, 1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 2, -1, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 2, 0, 1, 0, 1, xid, yid, zid, nx, ny, n_cells));
  test_data.emplace_back(
      mhd::_internal::_ctSlope(flux.data(), conserved.data(), -1, 2, 0, -1, 0, -1, xid, yid, zid, nx, ny, n_cells));

  // Check the results
  ASSERT_EQ(test_data.size(), fiducial_data.size());

  for (size_t i = 0; i < test_data.size(); i++) {
    testingUtilities::Check_Results(fiducial_data.at(i), test_data.at(i), "");
  }
}
// =============================================================================
#endif  // MHD
