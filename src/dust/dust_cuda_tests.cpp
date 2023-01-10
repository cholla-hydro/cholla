/*!
* \file dust_cuda_tests.cpp
* \author Helena Richie (helenarichie@pitt.edu)
* \brief Test dust model functions
*
*/

// STL Includes
#include <iostream>
#include <string>
#include <math.h>

// External Includes
#include <gtest/gtest.h> // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global_cuda.h"
#include "../utils/gpu.hpp"
#include "../utils/testing_utilities.h"
#include "../dust/dust_cuda.h" // Include code to test

#ifdef DUST

TEST(tDUSTTestSputteringTimescale, CorrectInputExpectCorrectOutput) // test suite name, test name
{
    // Parameters
    Real YR_IN_S = 3.154e7;
    Real const k_test_n = 1;
    Real const k_test_T = pow(10, 5.0);

    Real const k_fiducial_num = 182565146.96398282;

    Real test_num = calc_tau_sp(k_test_n, k_test_T) / YR_IN_S; // yr

    double abs_diff;
    int64_t ulps_diff;

    bool is_true;

    is_true = testingUtilities::nearlyEqualDbl(k_fiducial_num, test_num, abs_diff, ulps_diff);
    
    EXPECT_TRUE(is_true)
            << "The fiducial value is:       " << k_fiducial_num << std::endl
            << "The test value is:           " << test_num << std::endl
            << "The absolute difference is:  " << abs_diff << std::endl
            << "The ULP difference is:       " << ulps_diff << std::endl;
}

TEST(tDUSTTestSputteringGrowthRate, CorrectInputExpectCorrectOutput) // test suite name, test name
{
    // Parameters
    Real YR_IN_S = 3.154e7;
    Real const k_test_tau_sp = 0.17e6; // kyr
    Real const k_test_d_dust = 1e-26 / DENSITY_UNIT; // sim units

    Real const k_fiducial_num = -2.6073835738056728;

    Real test_num = calc_dd_dt(k_test_d_dust, k_test_tau_sp);

    double abs_diff;
    int64_t ulps_diff;

    bool is_true;

    is_true = testingUtilities::nearlyEqualDbl(k_fiducial_num, test_num, abs_diff, ulps_diff);
    
    EXPECT_TRUE(is_true)
            << "The fiducial value is:       " << k_fiducial_num << std::endl
            << "The test value is:           " << test_num << std::endl
            << "The absolute difference is:  " << abs_diff << std::endl
            << "The ULP difference is:       " << ulps_diff << std::endl;
}

#endif // DUST