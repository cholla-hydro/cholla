/*!
 * \file gravity_system_tests.cpp
 * \author Evan Schneider (eschneider@pitt.edu)
 * \brief Contains all the system tests for the GRAVITY build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

// =============================================================================
// Test Suite: tGRAVITYSYSTEMSphericalCollapse
// =============================================================================
/*!
 * \defgroup tGRAVITYSYSTEMSphericalCollapse_CorrectInputExpectCorrectOutput
 * \brief Test spherical collapse with hydro + FFT gravity initial conditions
 *
 */
/// @{
TEST(tGRAVITYSYSTEMSphericalCollapse, CorrectInputExpectCorrectOutput)
{
  system_test::SystemTestRunner collapseTest;
  collapseTest.runTest();
}
/// @}
// =============================================================================
