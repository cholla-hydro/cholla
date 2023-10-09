/*!
 * \file mhd_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of mhd_utilities.h and mhd_utilities.cpp
 *
 */

// STL Includes
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../grid/grid3D.h"
#include "../utils/mhd_utilities.h"
#include "../utils/testing_utilities.h"

// =============================================================================
// Local helper functions
namespace
{
struct TestParams {
  double gamma = 5. / 3.;
  std::vector<double> density{8.4087201154e-100, 1.6756968986e2, 5.4882403847e100};
  std::vector<double> velocityX{7.0378624601e-100, 7.0829278656e2, 1.8800514112e100};
  std::vector<double> velocityY{7.3583469014e-100, 5.9283073464e2, 5.2725717864e100};
  std::vector<double> velocityZ{1.7182972216e-100, 8.8417748226e2, 1.5855352639e100};
  std::vector<double> momentumX{8.2340416681e-100, 8.1019429453e2, 5.5062596954e100};
  std::vector<double> momentumY{4.9924582299e-100, 7.1254780684e2, 6.5939640992e100};
  std::vector<double> momentumZ{3.6703192739e-100, 7.5676716066e2, 7.2115881803e100};
  std::vector<double> energy{3.0342082433e-100, 7.6976906577e2, 1.9487120853e100};
  std::vector<double> pressureGas{2.2244082909e-100, 8.6772951021e2, 6.7261085663e100};
  std::vector<double> pressureTotal{8.1704748693e-100, 2.6084125198e2, 1.8242151369e100};
  std::vector<double> magneticX{2.8568843801e-100, 9.2400807786e2, 2.1621115264e100};
  std::vector<double> magneticY{9.2900880344e-100, 8.0382409757e2, 6.6499532343e100};
  std::vector<double> magneticZ{9.5795678229e-100, 3.3284839263e2, 9.2337456649e100};
  std::vector<std::string> names{"Small number case", "Medium number case", "Large number case"};
};
}  // namespace
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::computeThermalEnergy function
// =============================================================================
/*!
 * \brief Test the mhd::utils::computeThermalEnergy function with the standard
 * set of parameters.
 *
 */
TEST(tMHDComputeThermalEnergy, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> energyMultiplier{1.0E85, 1.0E4, 1.0E105};
  std::vector<double> fiducialGasPressures{3.0342082433e-15, 6887152.1495634327, 1.9480412919836246e+205};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testGasPressure = mhd::utils::computeThermalEnergy(
        energyMultiplier.at(i) * parameters.energy.at(i), parameters.density.at(i), parameters.momentumX.at(i),
        parameters.momentumY.at(i), parameters.momentumZ.at(i), parameters.magneticX.at(i), parameters.magneticY.at(i),
        parameters.magneticZ.at(i), parameters.gamma);

    testing_utilities::Check_Results(fiducialGasPressures.at(i), testGasPressure, parameters.names.at(i));
  }
}
// =============================================================================
// End of tests for the mhd::utils::computeThermalEnergy function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::computeMagneticEnergy function
// =============================================================================
/*!
 * \brief Test the mhd::utils::computeMagneticEnergy function with the standard
 * set of parameters.
 *
 */
TEST(tMHDcomputeMagneticEnergy, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> energyMultiplier{1.0E85, 1.0E4, 1.0E105};
  std::vector<double> fiducialEnergy{0.0, 805356.08013056568, 6.7079331637514162e+201};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testMagneticEnergy = mhd::utils::computeMagneticEnergy(parameters.magneticX.at(i), parameters.magneticY.at(i),
                                                                parameters.magneticZ.at(i));

    testing_utilities::Check_Results(fiducialEnergy.at(i), testMagneticEnergy, parameters.names.at(i));
  }
}
// =============================================================================
// End of tests for the mhd::utils::computeMagneticEnergy function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::computeTotalPressure function
// =============================================================================
/*!
 * \brief Test the mhd::utils::computeTotalPressure function with the standard
 * set of parameters.
 *
 */
TEST(tMHDComputeTotalPressure, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialTotalPressures{9.9999999999999995e-21, 806223.80964077567, 6.7079331637514151e+201};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testTotalPressure = mhd::utils::computeTotalPressure(parameters.pressureGas.at(i), parameters.magneticX.at(i),
                                                              parameters.magneticY.at(i), parameters.magneticZ.at(i));

    testing_utilities::Check_Results(fiducialTotalPressures.at(i), testTotalPressure, parameters.names.at(i));
  }
}

/*!
 * \brief Test the mhd::utils::computeTotalPressure function with a the standard
 * set of parameters. Gas pressure has been multiplied and made negative to
 * generate negative total pressures
 *
 */
TEST(tMHDComputeTotalPressure, NegativePressureExpectAutomaticFix)
{
  TestParams parameters;
  std::vector<double> pressureMultiplier{1.0, -1.0e4, -1.0e105};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testTotalPressure = mhd::utils::computeTotalPressure(pressureMultiplier.at(i) * parameters.pressureGas.at(i),
                                                              parameters.magneticX.at(i), parameters.magneticY.at(i),
                                                              parameters.magneticZ.at(i));

    // I'm using the binary equality assertion here since in the case of
    // negative pressure the function should return exactly TINY_NUMBER
    EXPECT_EQ(TINY_NUMBER, testTotalPressure) << "Difference in " << parameters.names.at(i) << std::endl;
  }
}
// =============================================================================
// End of tests for the mhd::utils::computeTotalPressure function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::fastMagnetosonicSpeed function
// =============================================================================
/*!
 * \brief Test the mhd::utils::fastMagnetosonicSpeed function with the standard
 * set of parameters. All values are reduced by 1e-25 in the large number case
 * to avoid overflow
 *
 */
TEST(tMHDFastMagnetosonicSpeed, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialFastMagnetosonicSpeed{1.9254472601190615e-40, 98.062482309387562, 1.5634816865472293e+38};
  std::vector<double> coef{1.0, 1.0, 1.0e-25};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testFastMagnetosonicSpeed = mhd::utils::fastMagnetosonicSpeed(
        coef.at(i) * parameters.density.at(i), coef.at(i) * parameters.pressureGas.at(i),
        coef.at(i) * parameters.magneticX.at(i), coef.at(i) * parameters.magneticY.at(i),
        coef.at(i) * parameters.magneticZ.at(i), parameters.gamma);

    testing_utilities::Check_Results(fiducialFastMagnetosonicSpeed.at(i), testFastMagnetosonicSpeed,
                                     parameters.names.at(i));
  }
}

/*!
 * \brief Test the mhd::utils::fastMagnetosonicSpeed function with the standard
 * set of parameters, density is negative. All values are reduced by 1e-25 in
 * the large number case to avoid overflow.
 *
 */
TEST(tMHDFastMagnetosonicSpeed, NegativeDensityExpectAutomaticFix)
{
  TestParams parameters;
  std::vector<double> fiducialFastMagnetosonicSpeed{1.9254472601190615e-40, 12694062010603.15, 1.1582688085027081e+86};
  std::vector<double> coef{1.0, 1.0, 1.0e-25};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testFastMagnetosonicSpeed = mhd::utils::fastMagnetosonicSpeed(
        -coef.at(i) * parameters.density.at(i), coef.at(i) * parameters.pressureGas.at(i),
        coef.at(i) * parameters.magneticX.at(i), coef.at(i) * parameters.magneticY.at(i),
        coef.at(i) * parameters.magneticZ.at(i), parameters.gamma);

    testing_utilities::Check_Results(fiducialFastMagnetosonicSpeed.at(i), testFastMagnetosonicSpeed,
                                     parameters.names.at(i));
  }
}
// =============================================================================
// End of tests for the mhd::utils::fastMagnetosonicSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::slowMagnetosonicSpeed function
// =============================================================================
/*!
 * \brief Test the mhd::utils::slowMagnetosonicSpeed function with the standard
 * set of parameters. All values are reduced by 1e-25 in the large number case
 * to avoid overflow
 *
 */
TEST(tMHDSlowMagnetosonicSpeed, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialSlowMagnetosonicSpeed{0.0, 2.138424778167535, 0.26678309355540852};
  // Coefficient to make sure the output is well defined and not nan or inf
  double const coef = 1E-95;

  for (size_t i = 2; i < parameters.names.size(); i++) {
    Real testSlowMagnetosonicSpeed = mhd::utils::slowMagnetosonicSpeed(
        parameters.density.at(i) * coef, parameters.pressureGas.at(i) * coef, parameters.magneticX.at(i) * coef,
        parameters.magneticY.at(i) * coef, parameters.magneticZ.at(i) * coef, parameters.gamma);

    testing_utilities::Check_Results(fiducialSlowMagnetosonicSpeed.at(i), testSlowMagnetosonicSpeed,
                                     parameters.names.at(i));
  }
}

/*!
 * \brief Test the mhd::utils::slowMagnetosonicSpeed function with the standard
 * set of parameters, density is negative. All values are reduced by 1e-25 in
 * the large number case to avoid overflow.
 *
 */
TEST(tMHDSlowMagnetosonicSpeed, NegativeDensityExpectAutomaticFix)
{
  TestParams parameters;
  std::vector<double> fiducialSlowMagnetosonicSpeed{0.0, 276816332809.37604, 1976400098318.3574};
  // Coefficient to make sure the output is well defined and not nan or inf
  double const coef = 1E-95;

  for (size_t i = 2; i < parameters.names.size(); i++) {
    Real testSlowMagnetosonicSpeed = mhd::utils::slowMagnetosonicSpeed(
        -parameters.density.at(i) * coef, parameters.pressureGas.at(i) * coef, parameters.magneticX.at(i) * coef,
        parameters.magneticY.at(i) * coef, parameters.magneticZ.at(i) * coef, parameters.gamma);

    testing_utilities::Check_Results(fiducialSlowMagnetosonicSpeed.at(i), testSlowMagnetosonicSpeed,
                                     parameters.names.at(i));
  }
}
// =============================================================================
// End of tests for the mhd::utils::slowMagnetosonicSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::alfvenSpeed function
// =============================================================================
/*!
 * \brief Test the mhd::utils::alfvenSpeed function with the standard set of
 * parameters.
 *
 */
TEST(tMHDAlfvenSpeed, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<double> fiducialAlfvenSpeed{2.8568843800999998e-90, 71.380245120271113, 9.2291462785524423e+49};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testAlfvenSpeed = mhd::utils::alfvenSpeed(parameters.magneticX.at(i), parameters.density.at(i));

    testing_utilities::Check_Results(fiducialAlfvenSpeed.at(i), testAlfvenSpeed, parameters.names.at(i));
  }
}

/*!
 * \brief Test the mhd::utils::alfvenSpeed function with the standard set of
 * parameters except density is negative
 *
 */
TEST(tMHDAlfvenSpeed, NegativeDensityExpectAutomaticFix)
{
  TestParams parameters;
  std::vector<double> fiducialAlfvenSpeed{2.8568843800999998e-90, 9240080778600, 2.1621115263999998e+110};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    Real testAlfvenSpeed = mhd::utils::alfvenSpeed(parameters.magneticX.at(i), -parameters.density.at(i));

    testing_utilities::Check_Results(fiducialAlfvenSpeed.at(i), testAlfvenSpeed, parameters.names.at(i));
  }
}
// =============================================================================
// End of tests for the mhd::utils::alfvenSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::cellCenteredMagneticFields function
// =============================================================================
#ifdef MHD
TEST(tMHDCellCenteredMagneticFields, CorrectInputExpectCorrectOutput)
{
  // Initialize the test grid and other state variables
  size_t const nx = 3, ny = nx;
  size_t const xid = std::floor(nx / 2), yid = xid, zid = xid;
  size_t const id = xid + yid * nx + zid * nx * ny;

  size_t const n_cells = std::pow(5, 3);
  // Make sure the vector is large enough that the locations where the
  // magnetic field would be in the real grid are filled
  std::vector<double> testGrid(n_cells * (grid_enum::num_fields));
  // Populate the grid with values where testGrid.at(i) = double(i). The
  // values chosen aren't that important, just that every cell has a unique
  // value
  std::iota(std::begin(testGrid), std::end(testGrid), 0.);

  // Fiducial and test variables
  double const fiducialAvgBx = 637.5, fiducialAvgBy = 761.5, fiducialAvgBz = 883.5;

  // Call the function to test
  auto [testAvgBx, testAvgBy, testAvgBz] =
      mhd::utils::cellCenteredMagneticFields(testGrid.data(), id, xid, yid, zid, n_cells, nx, ny);

  // Check the results
  testing_utilities::Check_Results(fiducialAvgBx, testAvgBx, "cell centered Bx value");
  testing_utilities::Check_Results(fiducialAvgBy, testAvgBy, "cell centered By value");
  testing_utilities::Check_Results(fiducialAvgBz, testAvgBz, "cell centered Bz value");
}
#endif  // MHD
// =============================================================================
// End of tests for the mhd::utils::cellCenteredMagneticFields function
// =============================================================================

// =============================================================================
// Tests for the mhd::utils::Init_Magnetic_Field_With_Vector_Potential function
// =============================================================================
#ifdef MHD
TEST(tMHDInitMagneticFieldWithVectorPotential, CorrectInputExpectCorrectOutput)
{
  // Mock up Header and Conserved structs
  Header H;
  Grid3D::Conserved C;

  H.nx      = 2;
  H.ny      = 2;
  H.nz      = 2;
  H.n_cells = H.nx * H.ny * H.nz;
  H.dx      = 0.2;
  H.dy      = 0.2;
  H.dz      = 0.2;

  double const default_fiducial = -999;
  std::vector<double> conserved_vector(H.n_cells * grid_enum::num_fields, default_fiducial);
  C.host       = conserved_vector.data();
  C.density    = &(C.host[grid_enum::density * H.n_cells]);
  C.momentum_x = &(C.host[grid_enum::momentum_x * H.n_cells]);
  C.momentum_y = &(C.host[grid_enum::momentum_y * H.n_cells]);
  C.momentum_z = &(C.host[grid_enum::momentum_z * H.n_cells]);
  C.Energy     = &(C.host[grid_enum::Energy * H.n_cells]);
  C.magnetic_x = &(C.host[grid_enum::magnetic_x * H.n_cells]);
  C.magnetic_y = &(C.host[grid_enum::magnetic_y * H.n_cells]);
  C.magnetic_z = &(C.host[grid_enum::magnetic_z * H.n_cells]);

  // Mock up vector potential
  std::vector<double> vector_potential(H.n_cells * 3, 0);
  std::iota(vector_potential.begin(), vector_potential.end(), 0);

  // Run the function
  mhd::utils::Init_Magnetic_Field_With_Vector_Potential(H, C, vector_potential);

  // Check the results
  double const bx_fiducial = -10.0;
  double const by_fiducial = 15.0;
  double const bz_fiducial = -5.0;

  for (size_t i = 0; i < conserved_vector.size(); i++) {
    if (i == 47) {
      testing_utilities::Check_Results(bx_fiducial, conserved_vector.at(i), "value at i = " + std::to_string(i));
    } else if (i == 55) {
      testing_utilities::Check_Results(by_fiducial, conserved_vector.at(i), "value at i = " + std::to_string(i));
    } else if (i == 63) {
      testing_utilities::Check_Results(bz_fiducial, conserved_vector.at(i), "value at i = " + std::to_string(i));
    } else {
      testing_utilities::Check_Results(default_fiducial, conserved_vector.at(i), "value at i = " + std::to_string(i));
    }
  }
}
#endif  // MHD
// =============================================================================
// End of tests for the mhd::utils::Init_Magnetic_Field_With_Vector_Potential function
// =============================================================================
