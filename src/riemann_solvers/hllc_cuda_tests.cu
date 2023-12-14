/*!
 * \file hllc_cuda_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Test the code units within hllc_cuda.cu
 *
 */

// STL Includes
#include <iostream>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global_cuda.h"
#include "../riemann_solvers/hllc_cuda.h"  // Include code to test
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"

#if defined(CUDA) && defined(HLLC)

// =========================================================================
/*!
 * \brief Test fixture for simple testing of the HLLC Riemann Solver.
   Effectively takes the left state, right state, fiducial fluxes, and
   custom user output then performs all the required running and testing
 *
 */
// NOLINTNEXTLINE(readability-identifier-naming)
class tHYDROCalculateHLLCFluxesCUDA : public ::testing::Test
{
 protected:
  // =====================================================================
  /*!
   * \brief Compute and return the HLLC fluxes
   *
   * \param[in] leftState The state on the left side in conserved
   * variables. In order the elements are: density, x-momentum,
   * y-momentum, z-momentum, and energy.
   * \param[in] rightState The state on the right side in conserved
   * variables. In order the elements are: density, x-momentum,
   * y-momentum, z-momentum, and energy.
   * \param[in] gamma The adiabatic index
   * \return std::vector<double>
   */
  std::vector<Real> Compute_Fluxes(std::vector<Real> const &stateLeft, std::vector<Real> const &stateRight,
                                   Real const &gamma)
  {
    // Simulation Paramters
    int const nx        = 1;  // Number of cells in the x-direction?
    int const ny        = 1;  // Number of cells in the y-direction?
    int const nz        = 1;  // Number of cells in the z-direction?
    int const nGhost    = 0;  // Isn't actually used it appears
    int const direction = 0;  // Which direction, 0=x, 1=y, 2=z
    int const nFields   = 5;  // Total number of conserved fields

    // Launch Parameters
    dim3 const dimGrid(1, 1, 1);   // How many blocks in the grid
    dim3 const dimBlock(1, 1, 1);  // How many threads per block

    // Create the std::vector to store the fluxes and declare the device
    // pointers
    std::vector<Real> testFlux(5);
    Real *devConservedLeft;
    Real *devConservedRight;
    Real *devTestFlux;

    // Allocate device arrays and copy data
    GPU_Error_Check(cudaMalloc(&devConservedLeft, nFields * sizeof(Real)));
    GPU_Error_Check(cudaMalloc(&devConservedRight, nFields * sizeof(Real)));
    GPU_Error_Check(cudaMalloc(&devTestFlux, nFields * sizeof(Real)));

    GPU_Error_Check(cudaMemcpy(devConservedLeft, stateLeft.data(), nFields * sizeof(Real), cudaMemcpyHostToDevice));
    GPU_Error_Check(cudaMemcpy(devConservedRight, stateRight.data(), nFields * sizeof(Real), cudaMemcpyHostToDevice));

    // Run kernel
    hipLaunchKernelGGL(Calculate_HLLC_Fluxes_CUDA, dimGrid, dimBlock, 0, 0,
                       devConservedLeft,   // the "left" interface
                       devConservedRight,  // the "right" interface
                       devTestFlux, nx, ny, nz, nGhost, gamma, direction, nFields);

    GPU_Error_Check();
    GPU_Error_Check(cudaMemcpy(testFlux.data(), devTestFlux, nFields * sizeof(Real), cudaMemcpyDeviceToHost));

    // Make sure to sync with the device so we have the results
    cudaDeviceSynchronize();
    GPU_Error_Check();

    return testFlux;
  }
  // =====================================================================

  // =====================================================================
  /*!
   * \brief Check if the fluxes are correct
   *
   * \param[in] fiducialFlux The fiducial flux in conserved variables. In
   * order the elements are: density, x-momentum, y-momentum, z-momentum,
   * and energy.
   * \param[in] testFlux The test flux in conserved variables. In order
   * the elements are: density, x-momentum, y-momentum, z-momentum, and
   * energy.
   * \param[in] customOutput Any custom output the user would like to
   * print. It will print after the default GTest output but before the
   * values that failed are printed
   */
  void Check_Results(std::vector<Real> const &fiducialFlux, std::vector<Real> const &testFlux,
                     std::string const &customOutput = "")
  {
    // Field names
    std::vector<std::string> const fieldNames{"Densities", "X Momentum", "Y Momentum", "Z Momentum", "Energies"};

    ASSERT_TRUE((fiducialFlux.size() == testFlux.size()) and (fiducialFlux.size() == fieldNames.size()))
        << "The fiducial flux, test flux, and field name vectors are not all "
           "the same length"
        << std::endl
        << "fiducialFlux.size() = " << fiducialFlux.size() << std::endl
        << "testFlux.size() = " << testFlux.size() << std::endl
        << "fieldNames.size() = " << fieldNames.size() << std::endl;

    // Check for equality
    for (size_t i = 0; i < fieldNames.size(); i++) {
      // Check for equality and if not equal return difference
      double absoluteDiff;
      int64_t ulpsDiff;

      bool areEqual = testing_utilities::nearlyEqualDbl(fiducialFlux[i], testFlux[i], absoluteDiff, ulpsDiff);
      EXPECT_TRUE(areEqual) << std::endl
                            << customOutput << std::endl
                            << "There's a difference in " << fieldNames[i] << " Flux" << std::endl
                            << "The fiducial value is:       " << fiducialFlux[i] << std::endl
                            << "The test value is:           " << testFlux[i] << std::endl
                            << "The absolute difference is:  " << absoluteDiff << std::endl
                            << "The ULP difference is:       " << ulpsDiff << std::endl;
    }
  }
  // =====================================================================
};
// =========================================================================

// =========================================================================
// Testing Calculate_HLLC_Fluxes_CUDA
/*!
* \brief Test the HLLC solver with the input from the high pressure side of a
sod shock tube. Correct results are hard coded into this test. Similar tests
do not need to be this verbose, simply passing values to the kernel call
should be sufficient in most cases
*
*/
TEST_F(tHYDROCalculateHLLCFluxesCUDA,        // Test suite name
       HighPressureSideExpectCorrectOutput)  // Test name
{
  // Physical Values
  Real const density   = 1.0;
  Real const pressure  = 1.0;
  Real const velocityX = 0.0;
  Real const velocityY = 0.0;
  Real const velocityZ = 0.0;
  Real const momentumX = density * velocityX;
  Real const momentumY = density * velocityY;
  Real const momentumZ = density * velocityZ;
  Real const gamma     = 1.4;
  Real const energy    = (pressure / (gamma - 1)) +
                      0.5 * density * (velocityX * velocityX + velocityY * velocityY + velocityZ * velocityZ);

  std::vector<Real> const state{density, momentumX, momentumY, momentumZ, energy};
  std::vector<Real> const fiducialFluxes{0, 1, 0, 0, 0};

  // Compute the fluxes
  std::vector<Real> const testFluxes = Compute_Fluxes(state,   // Left state
                                                      state,   // Right state
                                                      gamma);  // Adiabatic Index

  // Check for correctness
  Check_Results(fiducialFluxes, testFluxes);
}
// =========================================================================

#endif
