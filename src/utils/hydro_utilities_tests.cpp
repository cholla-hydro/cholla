/*!
 * \file hyo_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu), Helena Richie (helenarichie@pitt.edu)
 * \brief Tests for the contents of hydro_utilities.h and hydro_utilities.cpp
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/testing_utilities.h"
#include "../utils/hydro_utilities.h"
#include "../global/global.h"

// =============================================================================
// Local helper functions

/*!
* INDEX OF VARIABLES
* p : pressure
* vx, vy, vz : x, y, and z velocity
* d : density
* E : energy
* T : temperature
* px, py, pz : x, y, and z momentum
* n : number density
*/

namespace
{
    struct TestParams
    {
        double gamma = 5./3.;
        std::vector<double> d {8.4087201154e-100, 1.6756968986e2, 5.4882403847e100};
        std::vector<double> vx {7.0378624601e-100, 7.0829278656e2, 1.8800514112e100};
        std::vector<double> vy {7.3583469014e-100, 5.9283073464e2, 5.2725717864e100};
        std::vector<double> vz {1.7182972216e-100, 8.8417748226e2, 1.5855352639e100};
        std::vector<double> px {8.2340416681e-100, 8.1019429453e2, 5.5062596954e100};
        std::vector<double> py {4.9924582299e-100, 7.1254780684e2, 6.5939640992e100};
        std::vector<double> pz {3.6703192739e-100, 7.5676716066e2, 7.2115881803e100};
        std::vector<double> E {3.0342082433e-100, 7.6976906577e2, 1.9487120853e100};
        std::vector<double> p {2.2244082909e-100, 8.6772951021e2, 6.7261085663e100};
        std::vector<std::string> names{"Small number case", "Medium number case", "Large number case"};
    };
}

TEST(tHYDROSYSTEMHydroUtilsCalcPressureConserved, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<double> fiducial_pressures{3.3366124363499995e-100, 9.9999999999999995e-21, 2.4282508238146436e+100};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real test_p = hydro_utilities::Calc_Pressure_Conserved(parameters.p.at(i), parameters.d.at(i), parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i), parameters.gamma);

        testingUtilities::checkResults(fiducial_pressures.at(i), test_p, parameters.names.at(i));
    }
}