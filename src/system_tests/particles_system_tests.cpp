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
class tPARTICLESSYSTEMSphericalCollapseParameterizedMpi
      :public
      ::testing::TestWithParam<size_t>
{
public:
   tPARTICLESSYSTEMSphericalCollapseParameterizedMpi()
      :collapseTest(true)
      {};
    ~tPARTICLESSYSTEMSphericalCollapseParameterizedMpi() = default;

protected:
    systemTest::SystemTestRunner collapseTest;
};

TEST_P(tPARTICLESSYSTEMSphericalCollapseParameterizedMpi,
       CorrectInputExpectCorrectOutput)
{
    collapseTest.numMpiRanks = GetParam();
    collapseTest.runTest();
}

INSTANTIATE_TEST_SUITE_P(CorrectInputExpectCorrectOutput,
                         tPARTICLESSYSTEMSphericalCollapseParameterizedMpi,
                         ::testing::Values(1, 2, 4));
// =============================================================================
