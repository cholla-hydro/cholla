/*!
 * \file hydro_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the HYDRO build type
 *
 */


// External Libraries and Headers
#include <gtest/gtest.h>
#include <cmath> // provides std:sin

// Local includes
#include "../system_tests/system_tester.h"
#include "../utils/testing_utilities.h"



#ifndef PI
#define PI 3.141592653589793
#endif


// =============================================================================
// Test Suite: tHYDROSYSTEMSodShockTube
// =============================================================================
/*!
 * \defgroup tHYDROSYSTEMSodShockTubeParameterizedMpi_CorrectInputExpectCorrectOutput
 * \brief Test the Sod Shock tube initial conditions as a parameterized test
 * with varying numbers of MPI ranks
 *
 */
/// @{
class tHYDROSYSTEMSodShockTubeParameterizedMpi
      :public
      ::testing::TestWithParam<size_t>
{
protected:
    systemTest::SystemTestRunner sodTest;
};

TEST_P(tHYDROSYSTEMSodShockTubeParameterizedMpi,
       CorrectInputExpectCorrectOutput)
{
    sodTest.numMpiRanks = GetParam();
    sodTest.runTest();
}

INSTANTIATE_TEST_SUITE_P(CorrectInputExpectCorrectOutput,
                         tHYDROSYSTEMSodShockTubeParameterizedMpi,
                         ::testing::Values(1, 2, 4));
/// @}
// =============================================================================

TEST(tHYDROSYSTEMConstant,
     CorrectInputExpectCorrectOutput)
{
  systemTest::SystemTestRunner testObject(false, false, false);

  testObject.launchCholla();

  testObject.openHydroTestData();

  testingUtilities::analyticConstant(testObject,"density",1.0);
  testingUtilities::analyticConstant(testObject,"momentum_x",0.0);
  testingUtilities::analyticConstant(testObject,"momentum_y",0.0);
  testingUtilities::analyticConstant(testObject,"momentum_z",0.0);
  testingUtilities::analyticConstant(testObject,"Energy",1.5e-5);

}


TEST(tHYDROSYSTEMSoundWave3D,
     CorrectInputExpectCorrectOutput)
{
  double time = 0.05;
  double amplitude = 1e-5;
  double dx = 1./64.;
    
  double real_kx = 2*PI;//kx of the physical problem
  
  double kx = real_kx * dx;
  double speed = 1;//speed of wave is 1 since P = 0.6 and gamma = 1.666667
  double phase = kx*0.5 - speed * time * real_kx; //kx*0.5 for half-cell offset
  double tolerance = 1e-7;

  systemTest::SystemTestRunner testObject(false, false, false);

  testObject.launchCholla();

  testObject.openHydroTestData();

  testingUtilities::analyticSine(testObject,"density",1.0,amplitude,kx,0.0,0.0,phase,tolerance);
}
