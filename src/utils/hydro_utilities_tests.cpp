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
  std::vector<double> U_advected{1.3847303413e-10, 1.0756968986e2, 1.0882403847e100};
  std::vector<double> pressureTotal{8.1704748693e-100, 2.6084125198e2, 1.8242151369e100};
  std::vector<double> magnetic_x{2.8568843801e-100, 9.2400807786e2, 2.1621115264e100};
  std::vector<double> magnetic_y{9.2900880344e-100, 8.0382409757e2, 6.6499532343e100};
  std::vector<double> magnetic_z{9.5795678229e-100, 3.3284839263e2, 9.2337456649e100};
  std::vector<std::string> names{"Small number case", "Medium number case", "Large number case"};
};
}  // namespace

TEST(tHYDROtMHDHydroUtilsCalcPressurePrimitive, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_pressure{0, 139982878676.5015, 1.2697896247496674e+301};
#else   // not MHD
  std::vector<double> fiducial_pressure{1e-20, 139983415580.5549, 1.2697896247496674e+301};
#endif  // MHD

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ps = hydro_utilities::Calc_Pressure_Primitive(
        parameters.E.at(i), parameters.d.at(i), parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_pressure.at(i), test_Ps, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDHydroUtilsCalcPressureConserved, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_pressure{0, 139984067469.81754, 1.3965808056866668e+301};
#else   // not MHD
  std::vector<double> fiducial_pressure{1e-20, 139984604373.87094, 1.3965808056866668e+301};
#endif  // MHD

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_pressure = hydro_utilities::Calc_Pressure_Conserved(
        parameters.E.at(i), parameters.d.at(i), parameters.mx.at(i), parameters.my.at(i), parameters.mz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_pressure.at(i), test_pressure, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDHydroUtilsCalcPressurePrimitive, NegativePressureExpectAutomaticFix)
{
  TestParams parameters;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_pressure = hydro_utilities::Calc_Pressure_Primitive(
        parameters.E.at(i), parameters.d.at(i), 1E4 * parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    // I'm using the binary equality assertion here since in the case of
    // negative pressure the function should return exactly TINY_NUMBER
    EXPECT_EQ(TINY_NUMBER, test_pressure) << "Difference in " << parameters.names.at(i) << std::endl;
  }
}

TEST(tHYDROtMHDHydroUtilsCalcPressureConserved, NegativePressureExpectAutomaticFix)
{
  TestParams parameters;

  for (size_t i = 0; i < parameters.names.size() - 1; i++) {
    Real test_pressure = hydro_utilities::Calc_Pressure_Conserved(
        1E-10 * parameters.E.at(i), parameters.d.at(i), 1E4 * parameters.mx.at(i), 1E4 * parameters.my.at(i),
        1E4 * parameters.mz.at(i), parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i),
        parameters.magnetic_z.at(i));

    // I'm using the binary equality assertion here since in the case of
    // negative pressure the function should return exactly TINY_NUMBER
    EXPECT_EQ(TINY_NUMBER, test_pressure) << "Difference in " << parameters.names.at(i) << std::endl;
  }
}

TEST(tHYDROHydroUtilsCalcTemp, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ts{3465185.0560059389, 29370603.906644326, 28968949.83344138};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ts = hydro_utilities::Calc_Temp(parameters.P.at(i), parameters.n.at(i));

    testing_utilities::Check_Results(fiducial_Ts.at(i), test_Ts, parameters.names.at(i));
  }
}

#ifdef DE
TEST(tHYDROHydroUtilsCalcTempDE, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ts{5.123106988008801e-09, 261106139.02514684, 1.2105231166585662e+107};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ts =
        hydro_utilities::Calc_Temp_DE(parameters.d.at(i), parameters.ge.at(i), parameters.gamma, parameters.n.at(i));

    testing_utilities::Check_Results(fiducial_Ts.at(i), test_Ts, parameters.names.at(i));
  }
}
#endif  // DE

TEST(tHYDROtMHDHydroUtilsCalcEnergyPrimitive, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_energy{3.3366124363499997e-10, 2589863.8420712831, 1.9018677140549926e+300};
#else   // not MHD
  std::vector<double> fiducial_energy{3.3366124363499997e-10, 1784507.7619407175, 1.9018677140549926e+300};
#endif  // MHD

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Es = hydro_utilities::Calc_Energy_Primitive(
        parameters.P.at(i), parameters.d.at(i), parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_energy.at(i), test_Es, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDHydroUtilsCalcEnergyConserved, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_energy{3.3366124363499997e-10, 806673.86799851817, 6.7079331637514162e+201};
#else   // not MHD
  std::vector<double> fiducial_energy{3.3366124363499997e-10, 1317.7878679524658, 1.0389584427972784e+101};
#endif  // MHD

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Es = hydro_utilities::Calc_Energy_Conserved(
        parameters.P.at(i), parameters.d.at(i), parameters.mx.at(i), parameters.my.at(i), parameters.mz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_energy.at(i), test_Es, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDHydroUtilsCalcEnergyPrimitive, NegativePressureExpectAutomaticFix)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_energy{1.4999999999999998e-20, 2588562.2478059679, 1.9018677140549926e+300};
#else   // not MHD
  std::vector<double> fiducial_energy{0, 1783206.1676754025, 1.9018677140549926e+300};
#endif  // MHD
  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Es = hydro_utilities::Calc_Energy_Primitive(
        -parameters.P.at(i), parameters.d.at(i), parameters.vx.at(i), parameters.vy.at(i), parameters.vz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_energy.at(i), test_Es, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDHydroUtilsCalcEnergyConserved, NegativePressureExpectAutomaticFix)
{
  TestParams parameters;
#ifdef MHD
  std::vector<double> fiducial_energy{0, 805372.27373320318, 6.7079331637514162e+201};
#else   // not MHD
  std::vector<double> fiducial_energy{0, 16.193602637465997, 3.0042157852278494e+99};
#endif  // MHD
  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Es = hydro_utilities::Calc_Energy_Conserved(
        -parameters.P.at(i), parameters.d.at(i), parameters.mx.at(i), parameters.my.at(i), parameters.mz.at(i),
        parameters.gamma, parameters.magnetic_x.at(i), parameters.magnetic_y.at(i), parameters.magnetic_z.at(i));

    testing_utilities::Check_Results(fiducial_energy.at(i), test_Es, parameters.names.at(i));
  }
}

TEST(tHYDROHydroUtilsGetPressureFromDE, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducial_Ps{1.5927160260000002e-10, 71.713126573333341, 7.2549358980000001e+99};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real test_Ps = hydro_utilities::Get_Pressure_From_DE(parameters.E.at(i), parameters.U_total.at(i),
                                                         parameters.U_advected.at(i), parameters.gamma);

    testing_utilities::Check_Results(fiducial_Ps.at(i), test_Ps, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDCalcKineticEnergyFromVelocity, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialEnergies{0.0, 6.307524975350106e-145, 7.3762470327090601e+249};
  double const coef = 1E-50;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testEnergy = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(
        coef * parameters.d.at(i), coef * parameters.vx.at(i), coef * parameters.vy.at(i), coef * parameters.vz.at(i));

    testing_utilities::Check_Results(fiducialEnergies.at(i), testEnergy, parameters.names.at(i));
  }
}

TEST(tHYDROtMHDCalcKineticEnergyFromMomentum, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialEnergies{0.0, 0.0, 7.2568536478335773e+147};
  double const coef = 1E-50;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testEnergy = hydro_utilities::Calc_Kinetic_Energy_From_Momentum(
        coef * parameters.d.at(i), coef * parameters.mx.at(i), coef * parameters.my.at(i), coef * parameters.mz.at(i));

    testing_utilities::Check_Results(fiducialEnergies.at(i), testEnergy, parameters.names.at(i));
  }
}