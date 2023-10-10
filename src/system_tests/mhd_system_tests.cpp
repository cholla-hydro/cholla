/*!
 * \file mhd_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the MHD build type
 *
 */

// STL includes
#include <cmath>

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../io/io.h"
#include "../system_tests/system_tester.h"
#include "../utils/testing_utilities.h"

// =============================================================================
// Test Suite: tMHDSYSTEMLinearWavesParameterizedAngle
// =============================================================================
/*!
 * \defgroup tMHDSYSTEMLinearWavesParameterizedAngle
 * \brief Test the linear waves initial conditions as a parameterized test
 * with varying angles. Details in Gardiner & Stone 2008
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tMHDSYSTEMLinearWavesParameterizedAngle : public ::testing::TestWithParam<std::tuple<double, double, double, int>>
{
 public:
  tMHDSYSTEMLinearWavesParameterizedAngle() : waveTest(false, true, false, false){};

 protected:
  system_test::SystemTestRunner waveTest;
  inline static std::unordered_map<std::string, double> high_res_l2norms;

  void Set_Launch_Params(double const &waveSpeed, double const &rEigenVec_rho, double const &rEigenVec_MomentumX,
                         double const &rEigenVec_MomentumY, double const &rEigenVec_MomentumZ,
                         double const &rEigenVec_E, double const &rEigenVec_Bx, double const &rEigenVec_By,
                         double const &rEigenVec_Bz, double const &pitch, double const &yaw, double const &domain,
                         int const &domain_direction, double const &vx = 0.0, size_t const &N = 32)
  {
    // Constant for all tests
    double const gamma = 5. / 3.;
    double const tOut  = 2 * domain / waveSpeed;

    // Define vector values
    double x_len = domain, y_len = domain, z_len = domain;
    int nx = N, ny = N, nz = N;
    double vx_rot = vx, vy_rot = 0, vz_rot = 0;
    double Bx_rot = 1, By_rot = 1.5, Bz_rot = 0;

    double rEigenVec_Bx_rot = rEigenVec_Bx;
    double rEigenVec_By_rot = rEigenVec_By;
    double rEigenVec_Bz_rot = rEigenVec_Bz;

    double rEigenVec_MomentumX_rot = rEigenVec_MomentumX;
    double rEigenVec_MomentumY_rot = rEigenVec_MomentumY;
    double rEigenVec_MomentumZ_rot = rEigenVec_MomentumZ;

    switch (domain_direction) {
      case 1:
        x_len *= 2;
        nx *= 2;
        break;
      case 2:  // swap X and Y
        y_len *= 2;
        ny *= 2;
        break;
      case 3:  // swap X and Z
        z_len *= 2;
        nz *= 2;
        break;
      default:
        throw std::invalid_argument("Invalid value of domain_direction given to Set_Launch_Params");
        break;
    }

    // Settings
    waveTest.chollaLaunchParams.append(" nx=" + to_string_exact<int>(nx));
    waveTest.chollaLaunchParams.append(" ny=" + to_string_exact<int>(ny));
    waveTest.chollaLaunchParams.append(" nz=" + to_string_exact<int>(nz));
    waveTest.chollaLaunchParams.append(" tout=" + to_string_exact<double>(tOut));
    waveTest.chollaLaunchParams.append(" outstep=" + to_string_exact<double>(tOut));
    waveTest.chollaLaunchParams.append(" init=Linear_Wave");
    waveTest.chollaLaunchParams.append(" xmin=0.0");
    waveTest.chollaLaunchParams.append(" ymin=0.0");
    waveTest.chollaLaunchParams.append(" zmin=0.0");
    waveTest.chollaLaunchParams.append(" xlen=" + to_string_exact<double>(x_len));
    waveTest.chollaLaunchParams.append(" ylen=" + to_string_exact<double>(y_len));
    waveTest.chollaLaunchParams.append(" zlen=" + to_string_exact<double>(z_len));
    waveTest.chollaLaunchParams.append(" xl_bcnd=1");
    waveTest.chollaLaunchParams.append(" xu_bcnd=1");
    waveTest.chollaLaunchParams.append(" yl_bcnd=1");
    waveTest.chollaLaunchParams.append(" yu_bcnd=1");
    waveTest.chollaLaunchParams.append(" zl_bcnd=1");
    waveTest.chollaLaunchParams.append(" zu_bcnd=1");
    waveTest.chollaLaunchParams.append(" rho=1.0");
    waveTest.chollaLaunchParams.append(" vx=" + to_string_exact<double>(vx_rot));
    waveTest.chollaLaunchParams.append(" vy=" + to_string_exact<double>(vy_rot));
    waveTest.chollaLaunchParams.append(" vz=" + to_string_exact<double>(vz_rot));
    waveTest.chollaLaunchParams.append(" P=" + to_string_exact<double>(1 / gamma));
    waveTest.chollaLaunchParams.append(" Bx=" + to_string_exact<double>(Bx_rot));
    waveTest.chollaLaunchParams.append(" By=" + to_string_exact<double>(By_rot));
    waveTest.chollaLaunchParams.append(" Bz=" + to_string_exact<double>(Bz_rot));
    waveTest.chollaLaunchParams.append(" A='1e-6'");
    waveTest.chollaLaunchParams.append(" gamma=" + to_string_exact<double>(gamma));
    waveTest.chollaLaunchParams.append(" rEigenVec_rho=" + to_string_exact<double>(rEigenVec_rho));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumX=" + to_string_exact<double>(rEigenVec_MomentumX_rot));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumY=" + to_string_exact<double>(rEigenVec_MomentumY_rot));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumZ=" + to_string_exact<double>(rEigenVec_MomentumZ_rot));
    waveTest.chollaLaunchParams.append(" rEigenVec_E=" + to_string_exact<double>(rEigenVec_E));
    waveTest.chollaLaunchParams.append(" rEigenVec_Bx=" + to_string_exact<double>(rEigenVec_Bx_rot));
    waveTest.chollaLaunchParams.append(" rEigenVec_By=" + to_string_exact<double>(rEigenVec_By_rot));
    waveTest.chollaLaunchParams.append(" rEigenVec_Bz=" + to_string_exact<double>(rEigenVec_Bz_rot));
    waveTest.chollaLaunchParams.append(" pitch=" + to_string_exact<double>(pitch));
    waveTest.chollaLaunchParams.append(" yaw=" + to_string_exact<double>(yaw));
  }
};

// Fast Magnetosonic Waves Moving Left and Right
// =============================================
TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, FastMagnetosonicWaveRightMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 2.;
  std::vector<int> const numTimeSteps = {214, 204, 220};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 2;
  double const rEigenVec_MomentumX = prefix * 4;
  double const rEigenVec_MomentumY = prefix * -2;  // + for left wave
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * 4;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 9;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

// Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.2E-7, 5.4E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(6.5E-8, 6.5E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(6.11E-8, 5.5E-8);
#endif  // PCM

  high_res_l2norms["fast_" + std::to_string(domain_direction)] = waveTest.getL2Norm();
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, FastMagnetosonicWaveLeftMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 2.;
  std::vector<int> const numTimeSteps = {214, 204, 220};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 2;
  double const rEigenVec_MomentumX = prefix * -4;
  double const rEigenVec_MomentumY = prefix * 2;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * 4;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 9;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

// Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.2E-7, 5.4E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(6.5E-8, 6.5E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(6.1E-8, 5.5E-8);
#endif  // PCM
}

// Slow Magnetosonic Waves Moving Left and Right
// =============================================
TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, SlowMagnetosonicWaveRightMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 0.5;
  std::vector<int> const numTimeSteps = {854, 813, 880};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 4;
  double const rEigenVec_MomentumX = prefix * 2;
  double const rEigenVec_MomentumY = prefix * 4;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * -2;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 3;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(2.0E-8, 2.7E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.45E-9, 1.3E-9);
#endif  // PCM

  high_res_l2norms["slow_" + std::to_string(domain_direction)] = waveTest.getL2Norm();
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, SlowMagnetosonicWaveLeftMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 0.5;
  std::vector<int> const numTimeSteps = {854, 813, 880};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 4;
  double const rEigenVec_MomentumX = prefix * -2;
  double const rEigenVec_MomentumY = prefix * -4;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * -2;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 3;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(2.0E-8, 2.7E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.45E-9, 1.3E-9);
#endif  // PCM
}

// Alfven Waves Moving Left and Right
// =============================================
TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, AlfvenWaveRightMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 1.0;
  std::vector<int> const numTimeSteps = {427, 407, 440};

  double const rEigenVec_rho       = 0;
  double const rEigenVec_MomentumX = 0;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = -1;
  double const rEigenVec_Bx        = 0;
  double const rEigenVec_By        = 0;
  double const rEigenVec_Bz        = 1;
  double const rEigenVec_E         = 0;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(3.0E-8, 3.0E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.95e-09, 2.16e-09);
#endif  // PCM

  high_res_l2norms["alfven_" + std::to_string(domain_direction)] = waveTest.getL2Norm();
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, AlfvenWaveLeftMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 1.0;
  std::vector<int> const numTimeSteps = {427, 407, 440};

  double const rEigenVec_rho       = 0;
  double const rEigenVec_MomentumX = 0;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = 1;
  double const rEigenVec_Bx        = 0;
  double const rEigenVec_By        = 0;
  double const rEigenVec_Bz        = 1;
  double const rEigenVec_E         = 0;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(3.0E-8, 3.0E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.95e-09, 2.16e-09);
#endif  // PCM
}

// Contact Wave Moving Right
// ===================================
TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, MHDContactWaveCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed              = 1.0;
  std::vector<int> const numTimeSteps = {641, 620, 654};

  double const rEigenVec_rho       = 1;
  double const rEigenVec_MomentumX = 1;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = 0;
  double const rEigenVec_Bx        = 0;
  double const rEigenVec_By        = 0;
  double const rEigenVec_Bz        = 0;
  double const rEigenVec_E         = 0.5;
  double const velocityX           = waveSpeed;

  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction,
                    velocityX);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

// Check Results
// Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(5.4E-7, 5.4E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(3.0E-8, 3.0E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.41e-09, 1.5E-09);
#endif  // PCM

  high_res_l2norms["contact_" + std::to_string(domain_direction)] = waveTest.getL2Norm();
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, FastMagnetosonicWaveExpectSecondOrderConvergence)
{
  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Specific to this test
  double const waveSpeed              = 2.;
  std::vector<int> const numTimeSteps = {107, 102, 110};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 2;
  double const rEigenVec_MomentumX = prefix * 4;
  double const rEigenVec_MomentumY = prefix * -2;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * 4;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 9;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction, 0.0,
                    16);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Run the wave
  waveTest.runL1ErrorTest(3.0E-8, 4.0E-8);

  // Check the scaling
  double const low_res_l2norm = waveTest.getL2Norm();
  testing_utilities::Check_Results(4.0, low_res_l2norm / high_res_l2norms["fast_" + std::to_string(domain_direction)],
                                   "", 0.17);
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, SlowMagnetosonicWaveExpectSecondOrderConvergence)
{
  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Specific to this test
  double const waveSpeed              = 0.5;
  std::vector<int> const numTimeSteps = {427, 407, 440};

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 4;
  double const rEigenVec_MomentumX = prefix * 2;
  double const rEigenVec_MomentumY = prefix * 4;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * -2;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 3;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction, 0.0,
                    16);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Run the wave
  waveTest.runL1ErrorTest(3.0E-8, 4.0E-8);

  // Check the scaling
  double const low_res_l2norm = waveTest.getL2Norm();
  testing_utilities::Check_Results(4.0, low_res_l2norm / high_res_l2norms["slow_" + std::to_string(domain_direction)],
                                   "", 0.17);
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, AlfvenWaveExpectSecondOrderConvergence)
{
  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Specific to this test
  double const waveSpeed              = 1.0;
  std::vector<int> const numTimeSteps = {214, 204, 220};

  double const rEigenVec_rho       = 0;
  double const rEigenVec_MomentumX = 0;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = -1;
  double const rEigenVec_Bx        = 0;
  double const rEigenVec_By        = 0;
  double const rEigenVec_Bz        = 1;
  double const rEigenVec_E         = 0;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction, 0.0,
                    16);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Run the wave
  waveTest.runL1ErrorTest(3.0E-8, 4.0E-8);

  // Check the scaling
  double const low_res_l2norm = waveTest.getL2Norm();
  testing_utilities::Check_Results(4.0, low_res_l2norm / high_res_l2norms["alfven_" + std::to_string(domain_direction)],
                                   "", 0.17);
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedAngle, MHDContactWaveExpectSecondOrderConvergence)
{
  // Get the test parameters
  auto [pitch, yaw, domain, domain_direction] = GetParam();

  // Specific to this test
  double const waveSpeed              = 1.0;
  std::vector<int> const numTimeSteps = {321, 310, 327};

  double const rEigenVec_rho       = 1;
  double const rEigenVec_MomentumX = 1;
  double const rEigenVec_MomentumY = 0;
  double const rEigenVec_MomentumZ = 0;
  double const rEigenVec_Bx        = 0;
  double const rEigenVec_By        = 0;
  double const rEigenVec_Bz        = 0;
  double const rEigenVec_E         = 0.5;
  double const velocityX           = waveSpeed;

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz, pitch, yaw, domain, domain_direction,
                    velocityX, 16);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps[domain_direction - 1]);

  // Run the wave
  waveTest.runL1ErrorTest(3.0E-8, 4.0E-8);

  // Check the scaling
  double const low_res_l2norm = waveTest.getL2Norm();
  testing_utilities::Check_Results(
      4.0, low_res_l2norm / high_res_l2norms["contact_" + std::to_string(domain_direction)], "", 0.17);
}

INSTANTIATE_TEST_SUITE_P(, tMHDSYSTEMLinearWavesParameterizedAngle,
                         ::testing::Values(std::make_tuple(0.0 * M_PI, 0.0 * M_PI, 0.5, 1),
                                           std::make_tuple(0.0 * M_PI, 0.5 * M_PI, 0.5, 2),
                                           std::make_tuple(0.5 * M_PI, 0.0 * M_PI, 0.5, 3)
                                           // std::make_tuple(std::asin(2./3.),
                                           // std::asin(2./std::sqrt(5.)), 1.5, 1)
                                           ));
/// @}
// =============================================================================

// =============================================================================
// Test Suite: tMHDSYSTEMLinearWavesParameterizedMpi
// =============================================================================
/*!
 * \defgroup tMHDSYSTEMLinearWavesParameterizedMpi
 * \brief Test the linear waves initial conditions as a parameterized test
 * with varying numbers of MPI ranks. Details in Gardiner & Stone 2008
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tMHDSYSTEMLinearWavesParameterizedMpi : public ::testing::TestWithParam<int>
{
 public:
  tMHDSYSTEMLinearWavesParameterizedMpi() : waveTest(false, true, false, false){};

 protected:
  system_test::SystemTestRunner waveTest;

  void Set_Launch_Params(double const &waveSpeed, double const &rEigenVec_rho, double const &rEigenVec_MomentumX,
                         double const &rEigenVec_MomentumY, double const &rEigenVec_MomentumZ,
                         double const &rEigenVec_E, double const &rEigenVec_Bx, double const &rEigenVec_By,
                         double const &rEigenVec_Bz)
  {
    // Constant for all tests
    size_t const N      = 32;
    double const gamma  = 5. / 3.;
    double const domain = 0.5;
    double const tOut   = 2 * domain / waveSpeed;

    // Settings
    waveTest.chollaLaunchParams.append(" nx=" + to_string_exact<int>(2 * N));
    waveTest.chollaLaunchParams.append(" ny=" + to_string_exact<int>(N));
    waveTest.chollaLaunchParams.append(" nz=" + to_string_exact<int>(N));
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
    waveTest.chollaLaunchParams.append(" vx=0");
    waveTest.chollaLaunchParams.append(" vy=0");
    waveTest.chollaLaunchParams.append(" vz=0");
    waveTest.chollaLaunchParams.append(" P=" + to_string_exact<double>(1 / gamma));
    waveTest.chollaLaunchParams.append(" Bx=1");
    waveTest.chollaLaunchParams.append(" By=1.5");
    waveTest.chollaLaunchParams.append(" Bz=0");
    waveTest.chollaLaunchParams.append(" A='1e-6'");
    waveTest.chollaLaunchParams.append(" gamma=" + to_string_exact<double>(gamma));
    waveTest.chollaLaunchParams.append(" rEigenVec_rho=" + to_string_exact<double>(rEigenVec_rho));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumX=" + to_string_exact<double>(rEigenVec_MomentumX));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumY=" + to_string_exact<double>(rEigenVec_MomentumY));
    waveTest.chollaLaunchParams.append(" rEigenVec_MomentumZ=" + to_string_exact<double>(rEigenVec_MomentumZ));
    waveTest.chollaLaunchParams.append(" rEigenVec_E=" + to_string_exact<double>(rEigenVec_E));
    waveTest.chollaLaunchParams.append(" rEigenVec_Bx=" + to_string_exact<double>(rEigenVec_Bx));
    waveTest.chollaLaunchParams.append(" rEigenVec_By=" + to_string_exact<double>(rEigenVec_By));
    waveTest.chollaLaunchParams.append(" rEigenVec_Bz=" + to_string_exact<double>(rEigenVec_Bz));
  }
};

INSTANTIATE_TEST_SUITE_P(, tMHDSYSTEMLinearWavesParameterizedMpi, ::testing::Values(1, 2, 4));

// Slow Magnetosonic Waves Moving Left and Right
// =============================================
TEST_P(tMHDSYSTEMLinearWavesParameterizedMpi, SlowMagnetosonicWaveRightMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed = 0.5;
  int const numTimeSteps = 854;

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 4;
  double const rEigenVec_MomentumX = prefix * 2;
  double const rEigenVec_MomentumY = prefix * 4;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * -2;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 3;

  // Get the test parameters
  waveTest.numMpiRanks = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(2.0E-8, 2.7E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.4E-9, 1.3E-9);
#endif  // PCM
}

TEST_P(tMHDSYSTEMLinearWavesParameterizedMpi, SlowMagnetosonicWaveLeftMovingCorrectInputExpectCorrectOutput)
{
  // Specific to this test
  double const waveSpeed = 0.5;
  int const numTimeSteps = 854;

  double const prefix              = 1. / (2 * std::sqrt(5));
  double const rEigenVec_rho       = prefix * 4;
  double const rEigenVec_MomentumX = prefix * -2;
  double const rEigenVec_MomentumY = prefix * -4;
  double const rEigenVec_MomentumZ = prefix * 0;
  double const rEigenVec_Bx        = prefix * 0;
  double const rEigenVec_By        = prefix * -2;
  double const rEigenVec_Bz        = prefix * 0;
  double const rEigenVec_E         = prefix * 3;

  // Get the test parameters
  waveTest.numMpiRanks = GetParam();

  // Set the launch parameters
  Set_Launch_Params(waveSpeed, rEigenVec_rho, rEigenVec_MomentumX, rEigenVec_MomentumY, rEigenVec_MomentumZ,
                    rEigenVec_E, rEigenVec_Bx, rEigenVec_By, rEigenVec_Bz);

  // Set the number of timesteps
  waveTest.setFiducialNumTimeSteps(numTimeSteps);

  // Check Results. Values based on results in Gardiner & Stone 2008
#ifdef PCM
  waveTest.runL1ErrorTest(4.E-7, 4.E-7);
#elif defined(PLMC)
  waveTest.runL1ErrorTest(2.0E-8, 2.7E-8);
#elif defined(PPMC)
  waveTest.runL1ErrorTest(1.4E-9, 1.3E-9);
#endif  // PCM
}

/// @}
// =============================================================================

// =============================================================================
// Test Suite: tMHDSYSTEMParameterizedMpi
// =============================================================================
/*!
 * \defgroup tMHDSYSTEMParameterizedMpi
 * \brief Test initial conditions as a parameterized test with varying numbers of MPI ranks
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tMHDSYSTEMParameterizedMpi : public ::testing::TestWithParam<size_t>
{
 protected:
  system_test::SystemTestRunner test_runner;
};
INSTANTIATE_TEST_SUITE_P(, tMHDSYSTEMParameterizedMpi, ::testing::Values(1, 2, 4));

/// Test constant state with all magnetic fields set to zero
TEST_P(tMHDSYSTEMParameterizedMpi, ConstantWithZeroMagneticFieldCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test constant state with all magnetic fields set to one
TEST_P(tMHDSYSTEMParameterizedMpi, ConstantWithMagneticFieldCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test the MHD Einfeldt Strong Rarefaction (Einfeldt et al. 1991)
TEST_P(tMHDSYSTEMParameterizedMpi, EinfeldtStrongRarefactionCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test the Brio & Wu Shock Tube (Brio & Wu 1988)
TEST_P(tMHDSYSTEMParameterizedMpi, BrioAndWuShockTubeCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test the Dai & Woodward Shock Tube (Dai & Woodward 1998)
TEST_P(tMHDSYSTEMParameterizedMpi, DaiAndWoodwardShockTubeCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test the Ryu & Jones 1a Shock Tube (Ryu & Jones 1995)
TEST_P(tMHDSYSTEMParameterizedMpi, RyuAndJones1aShockTubeCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}

/// Test the Ryu & Jones 4d Shock Tube (Ryu & Jones 1995)
TEST_P(tMHDSYSTEMParameterizedMpi, RyuAndJones4dShockTubeCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  // This test is particularly sensitive to minor changes in the initial conditions, the kind of changes that are
  // expected from compiler to compiler. As such the limits have been loosened slightly.
  test_runner.setFixedEpsilon(7.3E-12);
  test_runner.runTest();
}

/// Test the Advecting Field Loop
TEST_P(tMHDSYSTEMParameterizedMpi, AdvectingFieldLoopCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();

  // Only do the L2 Norm test. The regular cell-to-cell comparison is brittle for this test across systems
  test_runner.runTest(true, 3.9E-8, 1.6E-6);
}

/// Test the MHD Blast Wave
TEST_P(tMHDSYSTEMParameterizedMpi, MhdBlastWaveCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();

  // Only do the L2 Norm test. The regular cell-to-cell comparison is brittle for this test across systems
  test_runner.runTest(true, 2.2E-4, 0.35);
}

/// Test the Orszag-Tang Vortex
TEST_P(tMHDSYSTEMParameterizedMpi, OrszagTangVortexCorrectInputExpectCorrectOutput)
{
  test_runner.numMpiRanks = GetParam();
  test_runner.runTest();
}
/// @}
// =============================================================================

// =============================================================================
// Test Suite: tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization
// =============================================================================
/*!
 * \defgroup tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization
 * \brief Test the circularly polarized Alfven Wave conditions as a parameterized test with varying polarizations.
 * Details in Gardiner & Stone 2008
 *
 */
/// @{
// NOLINTNEXTLINE(readability-identifier-naming)
class tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization : public ::testing::TestWithParam<double>
{
 public:
  tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization() : cpawTest(false, true, false, false){};

 protected:
  system_test::SystemTestRunner cpawTest;

  void Set_Launch_Params(double const &polarization, double const &vx)
  {
    // Constant for all tests
    size_t const N      = 32;
    double const length = 1.5;
    double const gamma  = 5. / 3.;
    double const tOut   = 1.0;
    double const pitch  = std::asin(2. / 3.);
    double const yaw    = std::asin(2. / std::sqrt(5.));

    // Domain settings
    double const x_len = 2. * length, y_len = length, z_len = length;
    int const nx = 2 * N, ny = N, nz = N;

    // Settings
    cpawTest.chollaLaunchParams.append(" nx=" + to_string_exact<int>(nx));
    cpawTest.chollaLaunchParams.append(" ny=" + to_string_exact<int>(ny));
    cpawTest.chollaLaunchParams.append(" nz=" + to_string_exact<int>(nz));
    cpawTest.chollaLaunchParams.append(" tout=" + to_string_exact<double>(tOut));
    cpawTest.chollaLaunchParams.append(" outstep=" + to_string_exact<double>(tOut));
    cpawTest.chollaLaunchParams.append(" init=Circularly_Polarized_Alfven_Wave");
    cpawTest.chollaLaunchParams.append(" xmin=0.0");
    cpawTest.chollaLaunchParams.append(" ymin=0.0");
    cpawTest.chollaLaunchParams.append(" zmin=0.0");
    cpawTest.chollaLaunchParams.append(" xlen=" + to_string_exact<double>(x_len));
    cpawTest.chollaLaunchParams.append(" ylen=" + to_string_exact<double>(y_len));
    cpawTest.chollaLaunchParams.append(" zlen=" + to_string_exact<double>(z_len));
    cpawTest.chollaLaunchParams.append(" xl_bcnd=1");
    cpawTest.chollaLaunchParams.append(" xu_bcnd=1");
    cpawTest.chollaLaunchParams.append(" yl_bcnd=1");
    cpawTest.chollaLaunchParams.append(" yu_bcnd=1");
    cpawTest.chollaLaunchParams.append(" zl_bcnd=1");
    cpawTest.chollaLaunchParams.append(" zu_bcnd=1");
    cpawTest.chollaLaunchParams.append(" polarization=" + to_string_exact<double>(polarization));
    cpawTest.chollaLaunchParams.append(" vx=" + to_string_exact<double>(vx));
    cpawTest.chollaLaunchParams.append(" gamma=" + to_string_exact<double>(gamma));
    cpawTest.chollaLaunchParams.append(" pitch=" + to_string_exact<double>(pitch));
    cpawTest.chollaLaunchParams.append(" yaw=" + to_string_exact<double>(yaw));
  }
};

// Moving wave with right and left polarization
// =============================================
TEST_P(tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization, MovingWaveCorrectInputExpectCorrectOutput)
{
  // Get the test parameter
  double const polarization = GetParam();

  // Set the wave to be moving
  double const vx = 0.0;

// Set allowed errors
#ifdef PCM
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(82);
  double const allowedL1Error = 6.5E-2;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 4.6E-2;
#elif defined(PLMC)
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(84);
  double const allowedL1Error = 5.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 5.0E-3;
#elif defined(PPMC)
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(84);
  double const allowedL1Error = 4.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 3.0E-3;
#elif defined(PLMP)
  double const allowedL1Error = 5.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 5.0E-3;
#elif defined(PPMP)
  double const allowedL1Error = 4.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 3.0E-3;
#endif

  // Set the launch parameters
  Set_Launch_Params(polarization, vx);

  // Check Results
  cpawTest.runL1ErrorTest(allowedL1Error, allowedError);
}

// Standing wave with right and left polarization
// =============================================
TEST_P(tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization, StandingWaveCorrectInputExpectCorrectOutput)
{
  // Get the test parameter
  double const polarization = GetParam();

  // Set the wave to be standing
  double const vx = -polarization;

// Set allowed errors
#ifdef PCM
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(130);
  double const allowedL1Error = 1.8E-2;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 1.7E-2;
#elif defined(PLMC)
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(130);
  double const allowedL1Error = 2.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 2.0E-3;
#elif defined(PPMC)
  // Set the number of timesteps
  cpawTest.setFiducialNumTimeSteps(130);
  double const allowedL1Error = 1.3E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 1.3E-3;
#elif defined(PLMP)
  double const allowedL1Error = 2.0E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 2.0E-3;
#elif defined(PPMP)
  double const allowedL1Error = 1.3E-3;  // Based on results in Gardiner & Stone 2008
  double const allowedError   = 1.3E-3;
#endif

  // Set the launch parameters
  Set_Launch_Params(polarization, vx);

  // Check Results
  cpawTest.runL1ErrorTest(allowedL1Error, allowedError);
}

INSTANTIATE_TEST_SUITE_P(, tMHDSYSTEMCircularlyPolarizedAlfvenWaveParameterizedPolarization,
                         ::testing::Values(1.0, -1.0));
/// @}
// =============================================================================
