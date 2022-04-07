/*!
 * \file mhd_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of mhd_utilities.h and mhd_utilities.cpp
 *
 */

// STL Includes
#include <vector>
#include <string>
#include <iostream>
#include <numeric>
#include <cmath>

// External Includes
#include <gtest/gtest.h>    // Include GoogleTest and related libraries/headers

// Local Includes
#include "../utils/testing_utilities.h"
#include "../utils/mhd_utilities.h"
#include "../global/global.h"

// =============================================================================
// Local helper functions
namespace
{
    struct testParams
    {
        double gamma = 5./3.;
        std::vector<double> density      {8.4087201154e-100, 1.6756968986e2, 5.4882403847e100};
        std::vector<double> velocityX    {7.0378624601e-100, 7.0829278656e2, 1.8800514112e100};
        std::vector<double> velocityY    {7.3583469014e-100, 5.9283073464e2, 5.2725717864e100};
        std::vector<double> velocityZ    {1.7182972216e-100, 8.8417748226e2, 1.5855352639e100};
        std::vector<double> momentumX    {8.2340416681e-100, 8.1019429453e2, 5.5062596954e100};
        std::vector<double> momentumY    {4.9924582299e-100, 7.1254780684e2, 6.5939640992e100};
        std::vector<double> momentumZ    {3.6703192739e-100, 7.5676716066e2, 7.2115881803e100};
        std::vector<double> energy       {3.0342082433e-100, 7.6976906577e2, 1.9487120853e100};
        std::vector<double> pressureGas  {2.2244082909e-100, 8.6772951021e2, 6.7261085663e100};
        std::vector<double> pressureTotal{8.1704748693e-100, 2.6084125198e2, 1.8242151369e100};
        std::vector<double> magneticX    {2.8568843801e-100, 9.2400807786e2, 2.1621115264e100};
        std::vector<double> magneticY    {9.2900880344e-100, 8.0382409757e2, 6.6499532343e100};
        std::vector<double> magneticZ    {9.5795678229e-100, 3.3284839263e2, 9.2337456649e100};
        std::vector<std::string> names{"Small number case", "Medium number case", "Large number case"};
    };
}
// =============================================================================


// =============================================================================
// Tests for the mhdUtils::computeEnergy function
// =============================================================================
/*!
 * \brief Test the mhdUtils::computeEnergy function with the standard set of
 * parameters
 *
 */
TEST(tMHDComputeEnergy,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> fiducialEnergies{3.3366124363499995e-100,
                                         137786230.15630624,
                                         9.2884430880010847e+301};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testEnergy = mhdUtils::computeEnergy(parameters.pressureGas.at(i),
                                                  parameters.density.at(i),
                                                  parameters.velocityX.at(i),
                                                  parameters.velocityY.at(i),
                                                  parameters.velocityZ.at(i),
                                                  parameters.magneticX.at(i),
                                                  parameters.magneticY.at(i),
                                                  parameters.magneticZ.at(i),
                                                  parameters.gamma);

        testingUtilities::checkResults(fiducialEnergies.at(i),
                                       testEnergy,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::computeEnergy function with a the standard set of
 * parameters except pressure is now negative
 *
 */
TEST(tMHDComputeEnergy,
     NegativePressureExpectAutomaticFix)
{
    testParams parameters;
    std::vector<double> fiducialEnergies{3.3366124363499995e-100,
                                         137784928.56204093,
                                         9.2884430880010847e+301};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testEnergy = mhdUtils::computeEnergy(-parameters.pressureGas.at(i),
                                                  parameters.density.at(i),
                                                  parameters.velocityX.at(i),
                                                  parameters.velocityY.at(i),
                                                  parameters.velocityZ.at(i),
                                                  parameters.magneticX.at(i),
                                                  parameters.magneticY.at(i),
                                                  parameters.magneticZ.at(i),
                                                  parameters.gamma);

        testingUtilities::checkResults(fiducialEnergies.at(i),
                                       testEnergy,
                                       parameters.names.at(i));
    }
}
// =============================================================================
// End of tests for the mhdUtils::computeEnergy function
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::computeGasPressure function
// =============================================================================
/*!
 * \brief Test the mhdUtils::computeGasPressure function with the standard set of
 * parameters. Energy has been increased to avoid negative pressures
 *
 */
TEST(tMHDComputeGasPressure,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> energyMultiplier{3, 1.0E4, 1.0E105};
    std::vector<double> fiducialGasPressures{1.8586864490415075e-100,
                                             4591434.7663756227,
                                             1.29869419465575e+205};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testGasPressure = mhdUtils::computeGasPressure(energyMultiplier.at(i) * parameters.energy.at(i),
                                                            parameters.density.at(i),
                                                            parameters.momentumX.at(i),
                                                            parameters.momentumY.at(i),
                                                            parameters.momentumZ.at(i),
                                                            parameters.magneticX.at(i),
                                                            parameters.magneticY.at(i),
                                                            parameters.magneticZ.at(i),
                                                            parameters.gamma);

        testingUtilities::checkResults(fiducialGasPressures.at(i),
                                       testGasPressure,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::computeGasPressure function with a the standard set
 * of parameters which produce negative pressures
 *
 */
TEST(tMHDComputeGasPressure,
     NegativePressureExpectAutomaticFix)
{
    testParams parameters;

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testGasPressure = mhdUtils::computeGasPressure(parameters.energy.at(i),
                                                            parameters.density.at(i),
                                                            parameters.momentumX.at(i),
                                                            parameters.momentumY.at(i),
                                                            parameters.momentumZ.at(i),
                                                            parameters.magneticX.at(i),
                                                            parameters.magneticY.at(i),
                                                            parameters.magneticZ.at(i),
                                                            parameters.gamma);

        // I'm using the binary equality assertion here since in the case of
        // negative pressure the function should return exactly TINY_NUMBER
        EXPECT_EQ(TINY_NUMBER, testGasPressure)
            << "Difference in " << parameters.names.at(i) << std::endl;
    }
}
// =============================================================================
// End of tests for the mhdUtils::computeGasPressure function
// =============================================================================


// =============================================================================
// Tests for the mhdUtils::computeThermalEnergy function
// =============================================================================
/*!
 * \brief Test the mhdUtils::computeThermalEnergy function with the standard set
 * of parameters.
 *
 */
TEST(tMHDComputeThermalEnergy,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> energyMultiplier{1.0E85, 1.0E4, 1.0E105};
    std::vector<double> fiducialGasPressures{3.0342082433e-15,
                                             6887152.1495634327,
                                             1.9480412919836246e+205};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testGasPressure = mhdUtils::computeThermalEnergy(energyMultiplier.at(i) * parameters.energy.at(i),
                                                              parameters.density.at(i),
                                                              parameters.momentumX.at(i),
                                                              parameters.momentumY.at(i),
                                                              parameters.momentumZ.at(i),
                                                              parameters.magneticX.at(i),
                                                              parameters.magneticY.at(i),
                                                              parameters.magneticZ.at(i),
                                                              parameters.gamma);

        testingUtilities::checkResults(fiducialGasPressures.at(i),
                                       testGasPressure,
                                       parameters.names.at(i));
    }
}
// =============================================================================
// End of tests for the mhdUtils::computeThermalEnergyfunction
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::computeTotalPressure function
// =============================================================================
/*!
 * \brief Test the mhdUtils::computeTotalPressure function with the standard set
 * of parameters.
 *
 */
TEST(tMHDComputeTotalPressure,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> fiducialTotalPressures{9.9999999999999995e-21,
                                               806223.80964077567,
                                               6.7079331637514151e+201};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testTotalPressure = mhdUtils::computeTotalPressure(parameters.pressureGas.at(i),
                                                                parameters.magneticX.at(i),
                                                                parameters.magneticY.at(i),
                                                                parameters.magneticZ.at(i));

        testingUtilities::checkResults(fiducialTotalPressures.at(i),
                                       testTotalPressure,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::computeTotalPressure function with a the standard
 * set of parameters. Gas pressure has been multiplied and made negative to
 * generate negative total pressures
 *
 */
TEST(tMHDComputeTotalPressure,
     NegativePressureExpectAutomaticFix)
{
    testParams parameters;
    std::vector<double> pressureMultiplier{1.0, -1.0e4, -1.0e105};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testTotalPressure = mhdUtils::computeTotalPressure(pressureMultiplier.at(i) * parameters.pressureGas.at(i),
                                                                parameters.magneticX.at(i),
                                                                parameters.magneticY.at(i),
                                                                parameters.magneticZ.at(i));

        // I'm using the binary equality assertion here since in the case of
        // negative pressure the function should return exactly TINY_NUMBER
        EXPECT_EQ(TINY_NUMBER, testTotalPressure)
            << "Difference in " << parameters.names.at(i) << std::endl;
    }
}
// =============================================================================
// End of tests for the mhdUtils::computeTotalPressure function
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::fastMagnetosonicSpeed function
// =============================================================================
/*!
 * \brief Test the mhdUtils::fastMagnetosonicSpeed function with the standard
 * set of parameters. All values are reduced by 1e-25 in the large number case
 * to avoid overflow
 *
 */
TEST(tMHDFastMagnetosonicSpeed,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> fiducialFastMagnetosonicSpeed{1.9254472601190615e-40,
                                                      98.062482309387562,
                                                      1.5634816865472293e+38};
    std::vector<double> coef{1.0, 1.0, 1.0e-25};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testFastMagnetosonicSpeed = mhdUtils::fastMagnetosonicSpeed(
                                                coef.at(i)*parameters.density.at(i),
                                                coef.at(i)*parameters.pressureGas.at(i),
                                                coef.at(i)*parameters.magneticX.at(i),
                                                coef.at(i)*parameters.magneticY.at(i),
                                                coef.at(i)*parameters.magneticZ.at(i),
                                                parameters.gamma);

        testingUtilities::checkResults(fiducialFastMagnetosonicSpeed.at(i),
                                       testFastMagnetosonicSpeed,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::fastMagnetosonicSpeed function with the standard
 * set of parameters, density is negative. All values are reduced by 1e-25 in
 * the large number case to avoid overflow.
 *
 */
TEST(tMHDFastMagnetosonicSpeed,
     NegativeDensityExpectAutomaticFix)
{
    testParams parameters;
    std::vector<double> fiducialFastMagnetosonicSpeed{1.9254472601190615e-40,
                                                      12694062010603.15,
                                                      1.1582688085027081e+86};
    std::vector<double> coef{1.0, 1.0, 1.0e-25};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testFastMagnetosonicSpeed = mhdUtils::fastMagnetosonicSpeed(
                                                -coef.at(i)*parameters.density.at(i),
                                                coef.at(i)*parameters.pressureGas.at(i),
                                                coef.at(i)*parameters.magneticX.at(i),
                                                coef.at(i)*parameters.magneticY.at(i),
                                                coef.at(i)*parameters.magneticZ.at(i),
                                                parameters.gamma);

        testingUtilities::checkResults(fiducialFastMagnetosonicSpeed.at(i),
                                       testFastMagnetosonicSpeed,
                                       parameters.names.at(i));
    }
}
// =============================================================================
// End of tests for the mhdUtils::fastMagnetosonicSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::slowMagnetosonicSpeed function
// =============================================================================
/*!
 * \brief Test the mhdUtils::slowMagnetosonicSpeed function with the standard
 * set of parameters. All values are reduced by 1e-25 in the large number case
 * to avoid overflow
 *
 */
TEST(tMHDSlowMagnetosonicSpeed,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> fiducialSlowMagnetosonicSpeed{0.0,
                                                      2.138424778167535,
                                                      0.26678309355540852};
    // Coefficient to make sure the output is well defined and not nan or inf
    double const coef = 1E-95;

    for (size_t i = 2; i < parameters.names.size(); i++)
    {
        Real testSlowMagnetosonicSpeed = mhdUtils::slowMagnetosonicSpeed(
                                                parameters.density.at(i) * coef,
                                                parameters.pressureGas.at(i) * coef,
                                                parameters.magneticX.at(i) * coef,
                                                parameters.magneticY.at(i) * coef,
                                                parameters.magneticZ.at(i) * coef,
                                                parameters.gamma);

        testingUtilities::checkResults(fiducialSlowMagnetosonicSpeed.at(i),
                                       testSlowMagnetosonicSpeed,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::slowMagnetosonicSpeed function with the standard
 * set of parameters, density is negative. All values are reduced by 1e-25 in
 * the large number case to avoid overflow.
 *
 */
TEST(tMHDSlowMagnetosonicSpeed,
     NegativeDensityExpectAutomaticFix)
{
    testParams parameters;
    std::vector<double> fiducialSlowMagnetosonicSpeed{0.0,
                                                      276816332809.37604,
                                                      1976400098318.3574};
    // Coefficient to make sure the output is well defined and not nan or inf
    double const coef = 1E-95;

    for (size_t i = 2; i < parameters.names.size(); i++)
    {
        Real testSlowMagnetosonicSpeed = mhdUtils::slowMagnetosonicSpeed(
                                                -parameters.density.at(i) * coef,
                                                parameters.pressureGas.at(i) * coef,
                                                parameters.magneticX.at(i) * coef,
                                                parameters.magneticY.at(i) * coef,
                                                parameters.magneticZ.at(i) * coef,
                                                parameters.gamma);

        testingUtilities::checkResults(fiducialSlowMagnetosonicSpeed.at(i),
                                       testSlowMagnetosonicSpeed,
                                       parameters.names.at(i));
    }
}
// =============================================================================
// End of tests for the mhdUtils::slowMagnetosonicSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::alfvenSpeed function
// =============================================================================
/*!
 * \brief Test the mhdUtils::alfvenSpeed function with the standard set of
 * parameters.
 *
 */
TEST(tMHDAlfvenSpeed,
     CorrectInputExpectCorrectOutput)
{
    testParams parameters;
    std::vector<double> fiducialAlfvenSpeed{2.8568843800999998e-90,
                                            71.380245120271113,
                                            9.2291462785524423e+49};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testAlfvenSpeed = mhdUtils::alfvenSpeed(parameters.magneticX.at(i),
                                                     parameters.density.at(i));

        testingUtilities::checkResults(fiducialAlfvenSpeed.at(i),
                                       testAlfvenSpeed,
                                       parameters.names.at(i));
    }
}

/*!
 * \brief Test the mhdUtils::alfvenSpeed function with the standard set of
 * parameters except density is negative
 *
 */
TEST(tMHDAlfvenSpeed,
     NegativeDensityExpectAutomaticFix)
{
    testParams parameters;
    std::vector<double> fiducialAlfvenSpeed{2.8568843800999998e-90,
                                            9240080778600,
                                            2.1621115263999998e+110};

    for (size_t i = 0; i < parameters.names.size(); i++)
    {
        Real testAlfvenSpeed = mhdUtils::alfvenSpeed(parameters.magneticX.at(i),
                                                     -parameters.density.at(i));

        testingUtilities::checkResults(fiducialAlfvenSpeed.at(i),
                                       testAlfvenSpeed,
                                       parameters.names.at(i));
    }
}
// =============================================================================
// End of tests for the mhdUtils::alfvenSpeed function
// =============================================================================

// =============================================================================
// Tests for the mhdUtils::cellCenteredMagneticFields function
// =============================================================================
TEST(tMHDCellCenteredMagneticFields,
     CorrectInputExpectCorrectOutput)
{
    // Initialize the test grid and other state variables
    size_t const nx = 3, ny = nx;
    size_t const xid = std::floor(nx/2), yid = xid, zid = xid;
    size_t const id = xid + yid*nx + zid*nx*ny;

    size_t const n_cells = std::pow(5,3);
    // Make sure the vector is large enough that the locations where the
    // magnetic field would be in the real grid are filled
    std::vector<double> testGrid(n_cells * (8+NSCALARS));
    // Populate the grid with values where testGrid.at(i) = double(i). The
    // values chosen aren't that important, just that every cell has a unique
    // value
    std::iota(std::begin(testGrid), std::end(testGrid), 0.);

    // Fiducial and test variables
    double const fiducialAvgBx = 637.5,
                 fiducialAvgBy = 761.5,
                 fiducialAvgBz = 883.5;
    double testAvgBx, testAvgBy, testAvgBz;

    // Call the function to test
    mhdUtils::cellCenteredMagneticFields(testGrid.data(), id, xid, yid, zid, n_cells, nx, ny, testAvgBx, testAvgBy, testAvgBz);

    // Check the results
    testingUtilities::checkResults(fiducialAvgBx, testAvgBx, "cell centered Bx value");
    testingUtilities::checkResults(fiducialAvgBy, testAvgBy, "cell centered By value");
    testingUtilities::checkResults(fiducialAvgBz, testAvgBz, "cell centered Bz value");
}
// =============================================================================
// End of tests for the mhdUtils::cellCenteredMagneticFields function
// =============================================================================
