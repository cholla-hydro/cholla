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
        hydro_utilities::Calc_Temp_DE(parameters.d.at(i) * parameters.ge.at(i), parameters.gamma, parameters.n.at(i));

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
  std::vector<double> fiducialEnergies{0.0, 6.307524975350106e-145, 1.9018677140549924e+150};
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
  std::vector<double> fiducialEnergies{0.0, 0.0, 3.0042157852278499e+49};
  double const coef = 1E-50;

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testEnergy = hydro_utilities::Calc_Kinetic_Energy_From_Momentum(
        coef * parameters.d.at(i), coef * parameters.mx.at(i), coef * parameters.my.at(i), coef * parameters.mz.at(i));

    testing_utilities::Check_Results(fiducialEnergies.at(i), testEnergy, parameters.names.at(i));
  }
}

TEST(tALLLoadCellPrimitive, CorrectInputExpectCorrectOutput)
{
  // Set up test and mock up grid
  size_t const nx = 3, ny = 3, nz = 3;
  size_t const n_cells = nx * ny * nz;
  size_t const xid = 1, yid = 1, zid = 1;
  size_t const o1 = grid_enum::momentum_x, o2 = grid_enum::momentum_y, o3 = grid_enum::momentum_z;
  Real const gamma = 5. / 3.;

  std::vector<Real> conserved(n_cells * grid_enum::num_fields);
  std::iota(conserved.begin(), conserved.end(), 0.0);

  // Up the energy part of the grid to avoid negative pressure
  for (size_t i = grid_enum::Energy * n_cells; i < (grid_enum::Energy + 1) * n_cells; i++) {
    conserved.at(i) *= 5.0E2;
  }

  for (int direction = 0; direction < 3; direction++) {
    // Get test data
    hydro_utilities::Primitive test_data;

    // Get the test data and permute the vector quantities back to the original order
    switch (direction) {
      case 0:
        test_data = hydro_utilities::Load_Cell_Primitive<0>(conserved.data(), xid, yid, zid, nx, ny, n_cells, gamma);
        break;
      case 1:
        test_data = hydro_utilities::Load_Cell_Primitive<1>(conserved.data(), xid, yid, zid, nx, ny, n_cells, gamma);
        math_utils::Cyclic_Permute_Twice(test_data.velocity);
#ifdef MHD
        math_utils::Cyclic_Permute_Twice(test_data.magnetic);
#endif  // MHD
        break;
      case 2:
        test_data = hydro_utilities::Load_Cell_Primitive<2>(conserved.data(), xid, yid, zid, nx, ny, n_cells, gamma);
        math_utils::Cyclic_Permute_Once(test_data.velocity);
#ifdef MHD
        math_utils::Cyclic_Permute_Once(test_data.magnetic);
#endif  // MHD
        break;
    }

// Check results
#ifdef MHD
    hydro_utilities::Primitive const fiducial_data{
        13, {3.0769230769230771, 5.1538461538461542, 7.2307692307692308}, 9662.3910256410272, {147.5, 173.5, 197.5}};
    testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
    testing_utilities::Check_Results(fiducial_data.velocity.x, test_data.velocity.x, "velocity.x");
    testing_utilities::Check_Results(fiducial_data.velocity.y, test_data.velocity.y, "velocity.y");
    testing_utilities::Check_Results(fiducial_data.velocity.z, test_data.velocity.z, "velocity.z");
    testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
    testing_utilities::Check_Results(fiducial_data.magnetic.x, test_data.magnetic.x, "magnetic.x");
    testing_utilities::Check_Results(fiducial_data.magnetic.y, test_data.magnetic.y, "magnetic.y");
    testing_utilities::Check_Results(fiducial_data.magnetic.z, test_data.magnetic.z, "magnetic.z");
#else  // MHD
    hydro_utilities::Primitive fiducial_data{
        13, {3.0769230769230771, 5.1538461538461542, 7.2307692307692308}, 39950.641025641031};
  #ifdef DE
    fiducial_data.pressure = 39950.641025641031;
  #endif  // DE
    testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
    testing_utilities::Check_Results(fiducial_data.velocity.x, test_data.velocity.x, "velocity.x");
    testing_utilities::Check_Results(fiducial_data.velocity.y, test_data.velocity.y, "velocity.y");
    testing_utilities::Check_Results(fiducial_data.velocity.z, test_data.velocity.z, "velocity.z");
    testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
#endif    // MHD
  }
}

TEST(tALLLoadCellConserved, CorrectInputExpectCorrectOutput)
{
  // Set up test and mock up grid
  size_t const nx = 3, ny = 3, nz = 3;
  size_t const n_cells = nx * ny * nz;
  size_t const xid = 1, yid = 1, zid = 1;
  size_t const o1 = grid_enum::momentum_x, o2 = grid_enum::momentum_y, o3 = grid_enum::momentum_z;
  Real const gamma = 5. / 3.;

  std::vector<Real> conserved(n_cells * grid_enum::num_fields);
  std::iota(conserved.begin(), conserved.end(), 0.0);

  // Up the energy part of the grid to avoid negative pressure
  for (size_t i = grid_enum::Energy * n_cells; i < (grid_enum::Energy + 1) * n_cells; i++) {
    conserved.at(i) *= 5.0E2;
  }

  for (int direction = 0; direction < 3; direction++) {
    // Get test data
    hydro_utilities::Conserved test_data;

    // Get the test data and permute the vector quantities back to the original order
    switch (direction) {
      case 0:
        test_data = hydro_utilities::Load_Cell_Conserved<0>(conserved.data(), xid, yid, zid, nx, ny, n_cells);
        break;
      case 1:
        test_data = hydro_utilities::Load_Cell_Conserved<1>(conserved.data(), xid, yid, zid, nx, ny, n_cells);
        math_utils::Cyclic_Permute_Twice(test_data.momentum);
#ifdef MHD
        math_utils::Cyclic_Permute_Twice(test_data.magnetic);
#endif  // MHD
        break;
      case 2:
        test_data = hydro_utilities::Load_Cell_Conserved<2>(conserved.data(), xid, yid, zid, nx, ny, n_cells);
        math_utils::Cyclic_Permute_Once(test_data.momentum);
#ifdef MHD
        math_utils::Cyclic_Permute_Once(test_data.magnetic);
#endif  // MHD
        break;
    }

    // Check results
    hydro_utilities::Conserved fiducial_data{13, {40, 67, 94}, 60500, {147.5, 173.5, 197.5}};
#ifdef MHD
    testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
    testing_utilities::Check_Results(fiducial_data.momentum.x, test_data.momentum.x, "momentum.x");
    testing_utilities::Check_Results(fiducial_data.momentum.y, test_data.momentum.y, "momentum.y");
    testing_utilities::Check_Results(fiducial_data.momentum.z, test_data.momentum.z, "momentum.z");
    testing_utilities::Check_Results(fiducial_data.energy, test_data.energy, "energy");
    testing_utilities::Check_Results(fiducial_data.magnetic.x, test_data.magnetic.x, "magnetic.x");
    testing_utilities::Check_Results(fiducial_data.magnetic.y, test_data.magnetic.y, "magnetic.y");
    testing_utilities::Check_Results(fiducial_data.magnetic.z, test_data.magnetic.z, "magnetic.z");
#else   // MHD
    testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
    testing_utilities::Check_Results(fiducial_data.momentum.x, test_data.momentum.x, "momentum.x");
    testing_utilities::Check_Results(fiducial_data.momentum.y, test_data.momentum.y, "momentum.y");
    testing_utilities::Check_Results(fiducial_data.momentum.z, test_data.momentum.z, "momentum.z");
    testing_utilities::Check_Results(fiducial_data.energy, test_data.energy, "energy");
#endif  // MHD
  }
}

TEST(tALLConserved2Primitive, CorrectInputExpectCorrectOutput)
{
  Real const gamma = 5. / 3.;
  hydro_utilities::Conserved input_data{2, {2, 3, 4}, 90, {6, 7, 8}, 9};
  hydro_utilities::Primitive test_data = hydro_utilities::Conserved_2_Primitive(input_data, gamma);

  hydro_utilities::Primitive fiducial_data{2, {1, 1.5, 2}, 55.166666666666671, {6, 7, 8}, 4.5};
#ifdef MHD
  fiducial_data.pressure = 5.5000000000000009;
#endif  // MHD

  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.velocity.x, test_data.velocity.x, "velocity.x");
  testing_utilities::Check_Results(fiducial_data.velocity.y, test_data.velocity.y, "velocity.y");
  testing_utilities::Check_Results(fiducial_data.velocity.z, test_data.velocity.z, "velocity.z");
  testing_utilities::Check_Results(fiducial_data.pressure, test_data.pressure, "pressure");
#ifdef MHD
  testing_utilities::Check_Results(fiducial_data.magnetic.x, test_data.magnetic.x, "magnetic.x");
  testing_utilities::Check_Results(fiducial_data.magnetic.y, test_data.magnetic.y, "magnetic.y");
  testing_utilities::Check_Results(fiducial_data.magnetic.z, test_data.magnetic.z, "magnetic.z");
#endif  // MHD
#ifdef DE
  testing_utilities::Check_Results(fiducial_data.gas_energy_specific, test_data.gas_energy_specific,
                                   "gas_energy_specific");
#endif  // DE
}

TEST(tALLPrimitive2Conserved, CorrectInputExpectCorrectOutput)
{
  Real const gamma = 5. / 3.;
  hydro_utilities::Primitive input_data{2, {2, 3, 4}, 90, {6, 7, 8}, 9};
  hydro_utilities::Conserved test_data = hydro_utilities::Primitive_2_Conserved(input_data, gamma);

  hydro_utilities::Conserved fiducial_data{2, {4, 6, 8}, 163.99999999999997, {6, 7, 8}, 18};
#ifdef MHD
  fiducial_data.energy = 238.49999999999997;
#endif  // MHD

  testing_utilities::Check_Results(fiducial_data.density, test_data.density, "density");
  testing_utilities::Check_Results(fiducial_data.momentum.x, test_data.momentum.x, "momentum.x");
  testing_utilities::Check_Results(fiducial_data.momentum.y, test_data.momentum.y, "momentum.y");
  testing_utilities::Check_Results(fiducial_data.momentum.z, test_data.momentum.z, "momentum.z");
  testing_utilities::Check_Results(fiducial_data.energy, test_data.energy, "energy");
#ifdef MHD
  testing_utilities::Check_Results(fiducial_data.magnetic.x, test_data.magnetic.x, "magnetic.x");
  testing_utilities::Check_Results(fiducial_data.magnetic.y, test_data.magnetic.y, "magnetic.y");
  testing_utilities::Check_Results(fiducial_data.magnetic.z, test_data.magnetic.z, "magnetic.z");
#endif  // MHD
#ifdef DE
  testing_utilities::Check_Results(fiducial_data.gas_energy, test_data.gas_energy, "gas_energy");
#endif  // DE
}