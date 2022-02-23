/*!
 * \file hydro_system_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for the HYDRO build type
 *
 */


// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h"

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