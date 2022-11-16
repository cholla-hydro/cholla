/*!
* \file hydro_cuda_tests.cu
* \author Evan Schneider (evs34@pitt.edu)
* \brief Test the code units within hydro_cuda.cu
*
*/

// STL Includes
#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"
#include "../utils/DeviceVector.h"
#include "../hydro/hydro_cuda.h"   // Include code to test

#if defined(CUDA)

// =============================================================================
// Tests for the Calc_dt_GPU function
// =============================================================================
TEST(tHYDROCalcDt3D, CorrectInputExpectCorrectOutput)
{
  // Call the function we are testing
  int num_blocks = 1;
  dim3 dim1dGrid(num_blocks, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);
  int const nx = 1;
  int const ny = 1;
  int const nz = 1;
  int const n_fields   = 5;  // Total number of conserved fields
  int const  n_ghost = 0;
  Real dx = 1.0;
  Real dy = 1.0;
  Real dz = 1.0;
  std::vector<Real> host_conserved(n_fields);
  cuda_utilities::DeviceVector<Real> dev_conserved(n_fields);
  cuda_utilities::DeviceVector<Real> dev_dti(1);
  Real gamma = 5.0/3.0;

  // Set values of conserved variables for input (host)
  host_conserved.at(0) = 1.0; // density
  host_conserved.at(1) = 0.0; // x momentum
  host_conserved.at(2) = 0.0; // y momentum
  host_conserved.at(3) = 0.0; // z momentum
  host_conserved.at(4) = 1.0; // Energy

  // Copy host data to device arrray
  CudaSafeCall(cudaMemcpy(dev_conserved, host_conserved, n_fields*sizeof(Real), cudaMemcpyHostToDevice));
  //__global__ void Calc_dt_3D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int n_fields, int nx, int ny, int nz, Real dx, Real dy, Real dz)

  // Run the kernel
  hipLaunchKernelGGL(Calc_dt_3D, dim1dGrid, dim1dBlock, 0, 0,
                     dev_conserved.data(), dev_dti.data(), gamma, n_ghost,
                     n_fields, nx, ny, nz, dx, dy, dz);
  CudaCheckError();

  // Compare results
  // Check for equality and if not equal return difference
  double const fiducialDt = 1.0540925533894598;
  double const testData = dev_dti.at(0);
  double absoluteDiff;
  int64_t ulpsDiff;
  bool areEqual;
  areEqual = testingUtilities::nearlyEqualDbl(fiducialDt, testData, absoluteDiff, ulpsDiff);
  EXPECT_TRUE(areEqual)
    << "The fiducial value is:       " << fiducialDt << std::endl
    << "The test value is:           " << testData     << std::endl
    << "The absolute difference is:  " << absoluteDiff << std::endl
    << "The ULP difference is:       " << ulpsDiff     << std::endl;
}
// =============================================================================
// End of tests for the Calc_dt_GPU function
// =============================================================================

// =============================================================================
// Tests for the hydroInverseCrossingTime function
// =============================================================================
TEST(tHYDROHydroInverseCrossingTime,
     CorrectInputExpectCorrectOutput)
{
// Set test values
double const energy    = 7.6976906577e2;
double const density   = 1.6756968986;
double const velocityX = 7.0829278656;
double const velocityY = 5.9283073464;
double const velocityZ = 8.8417748226;
double const cellSizeX = 8.1019429453e2;
double const cellSizeY = 7.1254780684e2;
double const cellSizeZ = 7.5676716066e2;
double const gamma = 5./3.;

// Fiducial Values
double const fiducialInverseCrossingTime = 0.038751126881804446;

// Function to test
double testInverseCrossingTime = hydroInverseCrossingTime(energy,
                                                         density,
                                                         1./density,
                                                         velocityX,
                                                         velocityY,
                                                         velocityZ,
                                                         cellSizeX,
                                                         cellSizeY,
                                                         cellSizeZ,
                                                         gamma);

// Check results
testingUtilities::checkResults(fiducialInverseCrossingTime, testInverseCrossingTime, "inverse crossing time");
}
// =============================================================================
// End of tests for the hydroInverseCrossingTime function
// =============================================================================

// =============================================================================
// Tests for the mhdInverseCrossingTime function
// =============================================================================
TEST(tMHDMhdInverseCrossingTime,
     CorrectInputExpectCorrectOutput)
{
  // Set test values
  double const energy    = 7.6976906577e2;
  double const density   = 1.6756968986;
  double const velocityX = 7.0829278656;
  double const velocityY = 5.9283073464;
  double const velocityZ = 8.8417748226;
  double const magneticX = 9.2400807786;
  double const magneticY = 8.0382409757;
  double const magneticZ = 3.3284839263;
  double const cellSizeX = 8.1019429453e2;
  double const cellSizeY = 7.1254780684e2;
  double const cellSizeZ = 7.5676716066e2;
  double const gamma = 5./3.;

  // Fiducial Values
  double const fiducialInverseCrossingTime = 0.038688028391959103;

  // Function to test
  double testInverseCrossingTime = mhdInverseCrossingTime(energy,
                                                          density,
                                                          1./density,
                                                          velocityX,
                                                          velocityY,
                                                          velocityZ,
                                                          magneticX,
                                                          magneticY,
                                                          magneticZ,
                                                          cellSizeX,
                                                          cellSizeY,
                                                          cellSizeZ,
                                                          gamma);


  // Check results
  testingUtilities::checkResults(fiducialInverseCrossingTime, testInverseCrossingTime, "inverse crossing time");
}
// =============================================================================
// End of tests for the mhdInverseCrossingTime function
// =============================================================================

#endif  // CUDA
