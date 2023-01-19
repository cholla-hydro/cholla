/*!
 * \file hydro_system_tests.cpp
 * \brief Contains all the system tests for the HYDRO build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

#include <cmath>  // provides std:sin

// Local includes
#include "../system_tests/system_tester.h"
#include "../utils/testing_utilities.h"

#ifndef PI
  #define PI 3.141592653589793
#endif

#define COOL_RHO 6.9498489284711

TEST(tCOOLINGSYSTEMConstant5, CorrectInputExpectCorrectOutput)
{
  // dt = 0.3
  // rho = COOL_RHO*1e5
  // pressure = 1e-3
  // T = 1e5
  /*
  double energy = 0.0014850544057189395;// Python
  */
  double energy = 0.00148501098087863;  // Cholla
  systemTest::SystemTestRunner testObject(false, false, false);
  testObject.launchCholla();
  testObject.openHydroTestData();

  testingUtilities::analyticConstant(testObject, "density", COOL_RHO * 1e5);
  testingUtilities::analyticConstant(testObject, "momentum_x", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_y", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_z", 0.0);
  testingUtilities::analyticConstant(testObject, "Energy", energy);
}

TEST(tCOOLINGSYSTEMConstant7, CorrectInputExpectCorrectOutput)
{
  // dt = 100
  // rho = COOL_RHO*1e5
  // pressure = 1e-1
  // T = 1e7
  // double energy = 0.14982743570299709; // Python
  double energy = 0.14982745510047499;  // Cholla
  systemTest::SystemTestRunner testObject(false, false, false);
  testObject.launchCholla();
  testObject.openHydroTestData();

  testingUtilities::analyticConstant(testObject, "density", COOL_RHO * 1e5);
  testingUtilities::analyticConstant(testObject, "momentum_x", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_y", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_z", 0.0);
  testingUtilities::analyticConstant(testObject, "Energy", energy);
}

TEST(tCOOLINGSYSTEMConstant8, CorrectInputExpectCorrectOutput)
{
  // dt = 90
  // rho = COOL_RHO*1e5
  // pressure = 1
  // T = 1e8

  // double energy = 1.499669522009355; // Python
  double energy = 1.4996695198095711;  // Cholla
  systemTest::SystemTestRunner testObject(false, false, false);
  testObject.launchCholla();
  testObject.openHydroTestData();

  testingUtilities::analyticConstant(testObject, "density", COOL_RHO * 1e5);
  testingUtilities::analyticConstant(testObject, "momentum_x", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_y", 0.0);
  testingUtilities::analyticConstant(testObject, "momentum_z", 0.0);
  testingUtilities::analyticConstant(testObject, "Energy", energy);
}
