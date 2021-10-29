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

TEST(tHYDROSYSTEMSodShockTube,
     CorrectInputExpectCorrectOutput)
{
    systemTest::SystemTestRunner sodTest;
    sodTest.runTest();
}

TEST(tHYDROSYSTEMSodShockTube,
     CorrectInput2MpiRanksExpectCorrectOutput)
{
    systemTest::SystemTestRunner sodTest;
    sodTest.numMpiRanks = 2;
    sodTest.runTest();
}

TEST(tHYDROSYSTEMSodShockTube,
     CorrectInput4MpiRanksExpectCorrectOutput)
{
    systemTest::SystemTestRunner sodTest;
    sodTest.numMpiRanks = 4;
    sodTest.runTest();
}