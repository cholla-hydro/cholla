/*!
 * \file hydro_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the HYDRO build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

#include <cmath>  // provides std:sin

// Local includes
#include "../io/io.h"
#include "../system_tests/system_tester.h"
#include "../utils/testing_utilities.h"

// =============================================================================
// Test Suite: tHYDROtMHDSYSTEMSodShockTube
// =============================================================================
/*!
 * \defgroup
 * tHYDROtMHDSYSTEMSodShockTubeParameterizedMpi_CorrectInputExpectCorrectOutput
 * \brief Test the Sod Shock tube initial conditions as a parameterized test
 * with varying numbers of MPI ranks
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tHYDROtMHDSYSTEMSodShockTubeParameterizedMpi : public ::testing::TestWithParam<size_t>
{
 protected:
  system_test::SystemTestRunner sodTest;
};

TEST_P(tHYDROtMHDSYSTEMSodShockTubeParameterizedMpi, CorrectInputExpectCorrectOutput)
{
#ifdef MHD
  sodTest.setFixedEpsilon(1.0E-4);

  // Don't test the gas energy fields
  auto datasetNames = sodTest.getDataSetsToTest();
  datasetNames.erase(std::remove(datasetNames.begin(), datasetNames.end(), "GasEnergy"), datasetNames.end());

  // Set the magnetic fiducial datasets to zero
  size_t const size = 64 * 64 * 65;
  std::vector<double> const magVec(size, 0);

  for (const auto *field : {"magnetic_x", "magnetic_y", "magnetic_z"}) {
    sodTest.setFiducialData(field, magVec);
    datasetNames.emplace_back(field);
  }

  sodTest.setDataSetsToTest(datasetNames);

  double const maxAllowedL1Error = 7.0E-3;
  double const maxAllowedError   = 4.6E-2;
#else
  double const maxAllowedL1Error = 9.4E-5;
  double const maxAllowedError   = 6.4E-4;
#endif  // MHD

  sodTest.numMpiRanks = GetParam();
  sodTest.runTest(true, maxAllowedL1Error, maxAllowedError);
}

INSTANTIATE_TEST_SUITE_P(CorrectInputExpectCorrectOutput, tHYDROtMHDSYSTEMSodShockTubeParameterizedMpi,
                         ::testing::Values(1, 2, 4));
/// @}
// =============================================================================

TEST(tHYDROSYSTEMSodShockTube, OneDimensionalCorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner sod_test;
  sod_test.runTest();
}

TEST(tHYDROSYSTEMSodShockTube, TwoDimensionalCorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner sod_test;
  sod_test.runTest();
}

TEST(tHYDROtMHDSYSTEMConstant, CorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner testObject(false, false, false);

  testObject.launchCholla();

  testObject.openHydroTestData();

  testing_utilities::analyticConstant(testObject, "density", 1.0);
  testing_utilities::analyticConstant(testObject, "momentum_x", 0.0);
  testing_utilities::analyticConstant(testObject, "momentum_y", 0.0);
  testing_utilities::analyticConstant(testObject, "momentum_z", 0.0);
  testing_utilities::analyticConstant(testObject, "Energy", 1.5e-5);
}

TEST(tHYDROtMHDSYSTEMSoundWave3D, CorrectInputExpectCorrectOutput)
{
  double time      = 0.05;
  double amplitude = 1e-5;
  double dx        = 1. / 64.;

  double real_kx = 2 * M_PI;  // kx of the physical problem

  double kx        = real_kx * dx;
  double speed     = 1;                                  // speed of wave is 1 since P = 0.6 and gamma = 1.666667
  double phase     = kx * 0.5 - speed * time * real_kx;  // kx*0.5 for half-cell offset
  double tolerance = 1e-7;

  system_test::SystemTestRunner testObject(false, false, false);

#ifdef MHD
  // Loosen correctness check to account for MHD only having PCM. This is
  // about the error between PCM and PPMP in hydro
  // Check Results. Values based on results in Gardiner & Stone 2008
  #ifdef PCM
  tolerance = 1e-6;
  #elif defined(PLMC)
  tolerance = 1.0E-7;
  #elif defined(PPMC)
  tolerance = 1.9E-9;
  #endif  // PCM
#endif    // MHD

  testObject.launchCholla();

  testObject.openHydroTestData();

  ASSERT_NO_FATAL_FAILURE(
      testing_utilities::analyticSine(testObject, "density", 1.0, amplitude, kx, 0.0, 0.0, phase, tolerance));
  ASSERT_NO_FATAL_FAILURE(
      testing_utilities::analyticSine(testObject, "momentum_x", 0.0, amplitude, kx, 0.0, 0.0, phase, tolerance));
  // testing_utilities::analyticSine(testObject,"momentum_y",0.0,amplitude,kx,0.0,0.0,0.0,tolerance);
  // testing_utilities::analyticSine(testObject,"momentum_z",0.0,amplitude,kx,0.0,0.0,0.0,tolerance);
}

// =============================================================================
// Test Suite: tHYDROtMHDSYSTEMLinearWavesParameterizedMpi
// =============================================================================
/*!
 * \defgroup tHYDROtMHDSYSTEMLinearWavesParameterizedMpi
 * \brief Test the linear waves initial conditions as a parameterized test
 * with varying numbers of MPI ranks.
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tHYDROtMHDSYSTEMLinearWavesParameterizedMpi : public ::testing::TestWithParam<size_t>
{
 public:
  tHYDROtMHDSYSTEMLinearWavesParameterizedMpi() : waveTest(false, true, false, false){};

 protected:
  system_test::SystemTestRunner waveTest;

#ifdef PCM
  double static constexpr allowedL1Error = 4E-7;  // Based on results in Gardiner & Stone 2008
  double static constexpr allowedError   = 4E-7;
#elif defined(PLMC)
  double static constexpr allowedL1Error = 1E-7;  // Based on results in Gardiner & Stone 2008
  double static constexpr allowedError   = 1E-7;
#elif defined(PLMP)
  double static constexpr allowedL1Error = 1E-7;  // Based on results in Gardiner & Stone 2008
  double static constexpr allowedError   = 1E-7;
#elif defined(PPMC)
  double static constexpr allowedL1Error = 2.7E-8;  // Based on results in Gardiner & Stone 2008
  double static constexpr allowedError   = 2.7E-8;
#elif defined(PPMP)
  double static constexpr allowedL1Error = 2.7E-8;  // Based on results in Gardiner & Stone 2008
  double static constexpr allowedError   = 2.7E-8;
#endif

  void Set_Launch_Params(double const &waveSpeed, double const &rEigenVec_rho, double const &rEigenVec_MomentumX,
                         double const &rEigenVec_MomentumY, double const &rEigenVec_MomentumZ,
                         double const &rEigenVec_E, double const &vx = 0.0)
  {
    // Constant for all tests
    size_t const N      = 32;
    double const domain = 0.5;
    double const gamma  = 5. / 3.;
    double const tOut   = 2 * domain / waveSpeed;

    // Settings
    waveTest.chollaLaunchParams.append(" nx=" + to_string_exact<double>(2 * N));
    waveTest.chollaLaunchParams.append(" ny=" + to_string_exact<double>(N));
    waveTest.chollaLaunchParams.append(" nz=" + to_string_exact<double>(N));
    waveTest.chollaLaunchParams.append(" tout=" + to_string_exact<double>(tOut));
    waveTest.chollaLaunchParams.append(" outstep=" + to_string_exact<double>(tOut));
    waveTest.chollaLaunchParams.append(" init=Linear_Wave");
    waveTest.chollaLaunchParams.append(" xmin=0.0");
    waveTest.chollaLaunchParams.append(" ymin=0.0");
    waveTest.chollaLaunchParams.append(" zmin=0.0");
    waveTest.chollaLaunchParams.append(" xlen=" + to_string_exact<double>(2 * domain));
    waveTest.chollaLaunchParams.append(" ylen=" + to_string_exact<double>(domain));
    waveTest.chollaLaunchParams.append(" zlen=" + to_string_exact<double>(domain));
    waveTest.chollaLaunchParams.append(" xl_bcnd=1");
    waveTest.chollaLaunchParams.append(" xu_bcnd=1");
    waveTest.chollaLaunchParams.append(" yl_bcnd=1");
    waveTest.chollaLaunchParams.append(" yu_bcnd=1");
    waveTest.chollaLaunchParams.append(" zl_bcnd=1");
    waveTest.chollaLaunchParams.append(" zu_bcnd=1");
    waveTest.chollaLaunchParams.append(" rho=1.0");
    waveTest.chollaLaunchParams.append(" vx=" + to_string_exact<double>(vx));
    waveTest.chollaLaunchParams.append(" vy=0");
    waveTest.chollaLaunchParams.append(" vz=0");
    waveTest.chollaLaunchParams.append(" P=" + to_string_exact<double>(1 / gamma));
    waveTest.chollaLaunchParams.append(" Bx=0");
    waveTest.chollaLaunchParams.append(" By=0");
    waveTest.chollaLaunchParams.append(" Bz=0");
    waveTest.chollaLaunchParams.append(" A='1e-6'");
    waveTest.chollaLaunchParams.append(" gamma=" + to_string_exact<double>(gamma));
    waveTest.chollaLaunchParams.append(" rEigenVec_rho=" + to_string_exact<double>(rEigenVec_rho));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumX=" + to_string_exact<double>(rEigenVec_MomentumX));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumY=" + to_string_exact<double>(rEigenVec_MomentumY));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumZ=" + to_string_exact<double>(rEigenVec_MomentumZ));
    waveTest.chollaLaunchParams.append(" rEigenVec_E=" + to_string_exact<double>(rEigenVec_E));
    waveTest.chollaLaunchParams.append(" rEigenVec_Bx=0");
    waveTest.chollaLaunchParams.append(" rEigenVec_By=0");
    waveTest.chollaLaunchParams.append(" rEigenVec_Bz=0");
  }
};

// Sound Waves Moving Left and Right
// =================================
TEST_P(tHYDROtMHDSYSTEMLinearWavesParameterizedMpi, SoundWaveRightMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed = 1.;
  int const numTimeSteps = 214;

  double const rEigenVec_rho       = 1;
  double const rEigenVec_MomentumX = 1;
  double const rEigenVec_MomentumY = 1;
  double const rEigenVec_MomentumZ = 1;
  double const rEigenVec_E         = 1.5;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E);

  // Set the number of MPI ranks
  waveTest.numMpiRanks = GetParam();

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps);

  // Check Results
  waveTest.runL1ErrorTest(2 * allowedL1Error, allowedError);
}

TEST_P(tHYDROtMHDSYSTEMLinearWavesParameterizedMpi, SoundWaveLeftMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed = 1.;
  int const numTimeSteps = 214;

  double const rEigenVec_rho       = 1;
  double const rEigenVec_MomentumX = -1;
  double const rEigenVec_MomentumY = 1;
  double const rEigenVec_MomentumZ = 1;
  double const rEigenVec_E         = 1.5;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E);

  // Set the number of MPI ranks
  waveTest.numMpiRanks = GetParam();

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps);

  // Check Results
  waveTest.runL1ErrorTest(2 * allowedL1Error, allowedError);
}

// Contact Waves Moving Left and Right
// ===================================
TEST_P(tHYDROtMHDSYSTEMLinearWavesParameterizedMpi, HydroContactWaveCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed = 1.0;
  int const numTimeSteps = 427;

  double const rEigenVec_rho       = 1;
  double const rEigenVec_MomentumX = 1;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = 0;
  double const rEigenVec_E         = 0.5;
  double const velocityX           = waveSpeed;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, velocityX);

  // Set the number of MPI ranks
  waveTest.numMpiRanks = GetParam();

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps);

  // Check Results
  waveTest.runL1ErrorTest(allowedL1Error, allowedError);
}

INSTANTIATE_TEST_SUITE_P(, tHYDROtMHDSYSTEMLinearWavesParameterizedMpi, ::testing::Values(1));
/// @}
// =============================================================================

TEST(tHYDROSYSTEMKhDiscontinuous, CorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner kh_discon_test;
  kh_discon_test.runTest();
}