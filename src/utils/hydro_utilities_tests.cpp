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

/*!
* INDEX OF VARIABLES
* P : pressure
* vx, vy, vz : x, y, and z velocity
* d : density
* E : energy
* T : temperature
* mx, my, mz : x, y, and z momentum
* n : number density
*/

// =============================================================================
// Local helper functions

namespace
{
    struct TestParams
    {
        double gamma = 5./3.;
        std::vector<double> d {1.0087201154e-100, 1.0756968986e2, 1.0882403847e100};
        std::vector<double> vx {1.0378624601e-100, 1.0829278656e2, 1.0800514112e100};
        std::vector<double> vy {1.0583469014e-100, 1.0283073464e2, 1.0725717864e100};
        std::vector<double> vz {1.0182972216e-100, 1.0417748226e2, 1.0855352639e100};
        std::vector<double> mx {0.2340416681e-100, 0.1019429453e2, 0.5062596954e100};
        std::vector<double> my {0.9924582299e-100, 0.1254780684e2, 0.5939640992e100};
        std::vector<double> mz {0.6703192739e-100, 0.5676716066e2, 0.2115881803e100};
        std::vector<double> E {20.9342082433e-90, 20.9976906577e10, 20.9487120853e300};
        std::vector<double> P {2.2244082909e-100, 8.6772951021e2, 6.7261085663e100};
        std::vector<double> n {1.0087201154e-100, 1.0756968986e2, 1.0882403847e100};
        std::vector<double> ge {1.0087201154e-100, 1.0756968986e2, 1.0882403847e100};
        std::vector<std::string> names{"Small number case", "Medium number case", "Large number case"};
    };
}

TEST(tHYDROSYSTEMHydroUtilsCalcPressurePrimitive, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<double> fiducial_Ps {1e-20, 139983415580.5549, 1.2697896247496674e+301};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real test_Ps = hydro_utilities::Calc_Pressure_Primitive(parameters.E.at(i), parameters.d.at(i), parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i), parameters.gamma);

        testingUtilities::checkResults(fiducial_Ps.at(i), test_Ps, parameters.names.at(i));
    }
}

TEST(tHYDROSYSTEMHydroUtilsCalcPressureConserved, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<double> fiducial_Ps {1e-20, 139984604373.87094, 1.3965808056866668e+301};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real test_Ps = hydro_utilities::Calc_Pressure_Conserved(parameters.E.at(i), parameters.d.at(i), parameters.mx.at(i), parameters.my.at(i), parameters.mz.at(i), parameters.gamma);

        testingUtilities::checkResults(fiducial_Ps.at(i), test_Ps, parameters.names.at(i));
    }
}

TEST(tHYDROSYSTEMHydroUtilsCalcTemp, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<double> fiducial_Ts {10335643.97340712, 37808388.612783447, 28968949.83344138};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real test_Ts = hydro_utilities::Calc_Temp(parameters.P.at(i), parameters.n.at(i));

        testingUtilities::checkResults(fiducial_Ts.at(i), test_Ts, parameters.names.at(i));
    }
}

#ifdef DE
TEST(tHYDROSYSTEMHydroUtilsCalcTempDE, CorrectInputExpectCorrectOutput) {
    TestParams parameters;
    std::vector<double> fiducial_Ts {10335643.97340712, 37808388.612783447, 28968949.83344138};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real test_Ts = hydro_utilities::Calc_Temp_DE(parameters.d.at(i), parameters.ge.at(i), parameters.gamma, parameters.n.at(i));

        testingUtilities::checkResults(fiducial_Ts.at(i), test_Ts, parameters.names.at(i));
    }
}
#endif // DE