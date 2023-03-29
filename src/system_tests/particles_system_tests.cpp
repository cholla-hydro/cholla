/*!
 * \file particles_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the PARTICLES build type
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

// =============================================================================
// Test Suite: tPARTICLESSYSTEMSphericalCollapse
// =============================================================================
/*!
 * \defgroup tPARTICLESSYSTEMSphericalCollapse_CorrectInputExpectCorrectOutput
 * \brief Test the spherical collapse with particles initial conditions
 *
 */
/// @{
TEST(tPARTICLESSYSTEMSphericalCollapse, DISABLED_CorrectInputExpectCorrectOutput)
{
  systemTest::SystemTestRunner collapseTest(true);
  collapseTest.runTest();
}
/// @}
// =============================================================================