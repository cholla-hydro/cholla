/*!
 * \file hyo_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu), Helena Richie
 * (helenarichie@pitt.edu) \brief Tests for the contents of hydro_utilities.h
 * and hydro_utilities.cpp
 *
 */

// STL Includes
#include <iostream>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

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
struct TestParams {
  double gamma = 5. / 3.;
  std::vector<double> d{1.0087201154e-15, 1.0756968986e2, 1.0882403847e100};
  std::vector<double> vx{1.0378624601e-100, 1.0829278656e2, 1.0800514112e100};
  std::vector<double> vy{1.0583469014e-100, 1.0283073464e2, 1.0725717864e100};
  std::vector<double> vz{1.0182972216e-100, 1.0417748226e2, 1.0855352639e100};
  std::vector<double> mx{0.2340416681e-100, 0.1019429453e2, 0.5062596954e100};
  std::vector<double> my{0.9924582299e-100, 0.1254780684e2, 0.5939640992e100};
  std::vector<double> mz{0.6703192739e-100, 0.5676716066e2, 0.2115881803e100};
  std::vector<double> E{20.9342082433e-90, 20.9976906577e10, 20.9487120853e300};
  std::vector<double> P{2.2244082909e-10, 8.6772951021e2, 6.7261085663e100};
  std::vector<double> n{3.0087201154e-10, 1.3847303413e2, 1.0882403847e100};
  std::vector<double> ge{4.890374019e-10, 1.0756968986e2, 3.8740982372e100};
  std::vector<double> U_total{2.389074039e-10, 4.890374019e2, 6.8731436293e100};
  std::vector<double> U_advected{1.3847303413e-10, 1.0756968986e2,
                                 1.0882403847e100};
  std::vector<std::string> names{"Small number case", "Medium number case",
                                 "Large number case"};
};
}  // namespace

TEST(tHYDROHydroUtilsCalcPressurePrimitive, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ps{1e-20, 139983415580.5549,
                                  1.2697896247496674e+301};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ps = hydro_utilities::Calc_Pressure_Primitive(
        parameters.E.at(i), parameters.d.at(i), parameters.vx.at(i),
        parameters.vy.at(i), parameters.vz.at(i), parameters.gamma);

    testingUtilities::checkResults(fiducial_Ps.at(i), test_Ps,
                                   parameters.names.at(i));
  }
}

TEST(tHYDROHydroUtilsCalcPressureConserved, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ps{1e-20, 139984604373.87094,
                                  1.3965808056866668e+301};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ps = hydro_utilities::Calc_Pressure_Conserved(
        parameters.E.at(i), parameters.d.at(i), parameters.mx.at(i),
        parameters.my.at(i), parameters.mz.at(i), parameters.gamma);

    testingUtilities::checkResults(fiducial_Ps.at(i), test_Ps,
                                   parameters.names.at(i));
  }
}

TEST(tHYDROHydroUtilsCalcTemp, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ts{3465185.0560059389, 29370603.906644326,
                                  28968949.83344138};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ts =
        hydro_utilities::Calc_Temp(parameters.P.at(i), parameters.n.at(i));

    testingUtilities::checkResults(fiducial_Ts.at(i), test_Ts,
                                   parameters.names.at(i));
  }
}

#ifdef DE
TEST(tHYDROHydroUtilsCalcTempDE, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ts{5.123106988008801e-09, 261106139.02514684,
                                  1.2105231166585662e+107};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ts =
        hydro_utilities::Calc_Temp_DE(parameters.d.at(i), parameters.ge.at(i),
                                      parameters.gamma, parameters.n.at(i));

    testingUtilities::checkResults(fiducial_Ts.at(i), test_Ts,
                                   parameters.names.at(i));
  }
}
#endif  // DE

TEST(tHYDROHydroUtilsCalcEnergyPrimitive, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Es{3.3366124363499997e-10, 1784507.7619407175,
                                  1.9018677140549926e+300};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Es = hydro_utilities::Calc_Energy_Primitive(
        parameters.P.at(i), parameters.d.at(i), parameters.vx.at(i),
        parameters.vy.at(i), parameters.vz.at(i), parameters.gamma);

    testingUtilities::checkResults(fiducial_Es.at(i), test_Es,
                                   parameters.names.at(i));
  }
}

TEST(tHYDROHydroUtilsGetPressureFromDE, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ps{1.5927160260000002e-10, 71.713126573333341,
                                  7.2549358980000001e+99};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ps = hydro_utilities::Get_Pressure_From_DE(
        parameters.E.at(i), parameters.U_total.at(i),
        parameters.U_advected.at(i), parameters.gamma);

    testingUtilities::checkResults(fiducial_Ps.at(i), test_Ps,
                                   parameters.names.at(i));
  }
}

TEST(tHYDROtMHDCalcKineticEnergyFromVelocity, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialEnergies{0.0, 6.307524975350106e-145,
                                       7.3762470327090601e+249};
  double const coef = 1E-50;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testEnergy = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(
        coef * parameters.d.at(i), coef * parameters.vx.at(i),
        coef * parameters.vy.at(i), coef * parameters.vz.at(i));

    testingUtilities::checkResults(fiducialEnergies.at(i), testEnergy,
                                   parameters.names.at(i));
  }
}

TEST(tHYDROtMHDCalcKineticEnergyFromMomentum, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialEnergies{0.0, 0.0, 7.2568536478335773e+147};
  double const coef = 1E-50;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testEnergy = hydro_utilities::Calc_Kinetic_Energy_From_Momentum(
        coef * parameters.d.at(i), coef * parameters.mx.at(i),
        coef * parameters.my.at(i), coef * parameters.mz.at(i));

    testingUtilities::checkResults(fiducialEnergies.at(i), testEnergy,
                                   parameters.names.at(i));
  }
}