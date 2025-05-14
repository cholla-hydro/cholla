/*!
 * \file cosmology_system_tests.cpp
 * \brief Contains all the system tests for the GRAVITY build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

TEST(tCOSMOLOGYSYSTEM50Mpc, CorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner cosmo(true, true, true, true, true);
  // we need to do the following to ensure the test passes (we are just maintaining
  // backwards compatability)
  cosmo.chollaLaunchParams.append(" w0=0.0");
  cosmo.runTest(true, 1.0e-07, 0.0006);
}
