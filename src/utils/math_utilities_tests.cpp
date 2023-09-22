/*!
 * \file math_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of math_utilities.h
 *
 */

// STL Includes
#include <math.h>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../utils/math_utilities.h"
#include "../utils/testing_utilities.h"

// =============================================================================
TEST(tALLRotateCoords, CorrectInputExpectCorrectOutput)
{
  // Fiducial values
  double const x_1         = 19.2497333410;
  double const x_2         = 60.5197699003;
  double const x_3         = 86.0613942621;
  double const pitch       = 1.239 * M_PI;
  double const yaw         = 0.171 * M_PI;
  double const x_1_rot_fid = -31.565679455456568;
  double const x_2_rot_fid = 14.745363873361605;
  double const x_3_rot_fid = -76.05402749550727;

  auto [x_1_rot, x_2_rot, x_3_rot] = math_utils::rotateCoords<double>(x_1, x_2, x_3, pitch, yaw);

  testing_utilities::checkResults<0>(x_1_rot_fid, x_1_rot, "x_1 rotated values");
  testing_utilities::checkResults<0>(x_2_rot_fid, x_2_rot, "x_2 rotated values");
  testing_utilities::checkResults<0>(x_3_rot_fid, x_3_rot, "x_3 rotated values");
}
// =============================================================================

// =========================================================================
/*!
 * \brief Test the math_utils::dotProduct function
 *
 */
TEST(tALLDotProduct, CorrectInputExpectCorrectOutput)
{
  std::vector<double> a{21.503067766457753, 48.316634031589935, 81.12177317622657},
      b{38.504606872151484, 18.984145880030045, 89.52561861038686};

  double const fiducialDotProduct = 9007.6941261535867;

  double testDotProduct;

  testDotProduct = math_utils::dotProduct(a.at(0), a.at(1), a.at(2), b.at(0), b.at(1), b.at(2));

  // Now check results
  testing_utilities::checkResults(fiducialDotProduct, testDotProduct, "dot product");
}
// =========================================================================