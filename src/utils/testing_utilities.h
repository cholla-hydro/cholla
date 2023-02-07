/*!
 * \file testing_utilites.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Header file for various testing related utility functions and the
 * globalChollaRoot global variable
 *
 */

#pragma once

// STL includes
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "../system_tests/system_tester.h"  // provide systemTest class

// =============================================================================
// NOTE: Global variables are declared as extern at the end of this file
// =============================================================================

/*!
 * \brief A namespace for various testing related utility functions. Many of
 * those functions will likely use the STL so this namespace should not be
 * considered compatible with CUDA/HIP.
 *
 */
namespace testingUtilities
{
// =========================================================================
/*!
 * \brief Compute the Units in the Last Place (ULP) difference between two
 * doubles
 *
 * \details This function is modified from
 * [Comparing Floating-Point Numbers Is Tricky by Matt
 * Kline](https://bitbashing.io/comparing-floats.html) which is in turn based on
 * [Comparing Floating Point Numbers, 2012 Edition by Bruce
 * Dawson](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
 * The latter seems to be the bible of floating point comparison and is the
 * basis of Googletests ASSERT_DOUBLE_EQ assertion.
 *
 * This particular function checks that the two numbers if the numbers are
 * perfectly equal, +0, -0, Nan, inf, or differently signed then it computes
 * the ULP difference between them are returns it
 *
 * \param[in] a The first double you wish to compare. Order doesn't matter.
 * \param[in] b The second double you wish to compare. Order doesn't matter.
 * \return int64_t The ULP distance between a and b.
 */
int64_t ulpsDistanceDbl(double const &a, double const &b);
// =========================================================================

// =========================================================================
/*!
 * \brief Check if two doubles are nearly equal.
 *
 * \details This function checks if two doubles are "nearly equal" which is
 * defined as either: A) the absolute difference between them is less than
 * the fixedEpsilon argument or B) the units in the last place (ULP)
 * difference is less  than the ulpsEpsilon argument. Both of the epsilon
 * arguments have default values which generally should not need to be
 * changed.
 *
 * Why does fixedEpsilon default to 1E-14? Running the Sod shock tube when
 * Cholla was compiled with GCC 9.3.0 vs. XL 16.1.1-10 on Summit lead to
 * absolute differences in the results up to 1.77636E-15. A priori we chose
 * that a difference between two numbers that was less than one order of
 * magnitude greater than the difference between compilers would be
 * considered "equal". I.e. since the maximum absolute error between the GCC
 * and XL compilers was ~1.7E-15 our allowed margin of error should be
 * ~1E-14.
 *
 * Why does ulpsEpsilon default to 4? Repeating the test above I computed
 * the largest ULP difference that wasn't caught by the absolute difference
 * requirement of 1E-14. It turns out that there were no uncaught
 * differences at all so I kept ulpsEpsilon at 4 since that's the Googletest
 * default for their floating point assertions
 *
 * \param[in] a The first double you wish to compare. Order doesn't matter.
 * \param[in] b The first double you wish to compare. Order doesn't matter.
 * \param[out] absoluteDiff The absolute difference between the numbers.
 * Only returned if the numbers are not equal. If the numbers are equal then
 * behaviour is undefined
 * \param[out] ulpsDiff The ULP difference between the numbers.
 * Only returned if the numbers are not equal. If the numbers are equal then
 * behaviour is undefined
 * \param[in] fixedEpsilon The allowed difference in real numbers. Defaults
 * to 1E-14
 * \param[in] ulpsEpsilon The allowed difference of ULPs. Defaults to 4
 * \return bool Whether or not the numbers are equal
 */
bool nearlyEqualDbl(double const &a, double const &b, double &absoluteDiff,
                    int64_t &ulpsDiff, double const &fixedEpsilon = 1E-14,
                    int64_t const &ulpsEpsilon = 4);
// =========================================================================

void wrapperEqual(int i, int j, int k, std::string dataSetName,
                  double test_value, double fid_value, double fixedEpsilon);

void analyticConstant(systemTest::SystemTestRunner testObject,
                      std::string dataSetName, double value);

void analyticSine(systemTest::SystemTestRunner testObject,
                  std::string dataSetName, double constant, double amplitude,
                  double kx, double ky, double kz, double phase,
                  double tolerance);

// =========================================================================
/*!
 * \brief A simple function to compare two doubles with the nearlyEqualDbl
 * function, perform a GTest assert on the result, and print out the values
 *
 * \tparam checkType The type of GTest assertion to use. "0" for and
 * "EXPECT" and "1" for an "ASSERT"
 * \param[in] fiducialNumber The fiducial number to test against
 * \param[in] testNumber The unverified number to test
 * \param[in] outString A string to be printed in the first line of the output
 * message. Format will be "Difference in outString"
 * \param[in] fixedEpsilon The fixed epsilon to use in the comparison.
 * Negative values are ignored and default behaviour is used
 * \param[in] ulpsEpsilon The ULP epsilon to use in the comparison. Negative
 * values are ignored and default behaviour is used
 */
template <int checkType = 0>
void checkResults(double fiducialNumber, double testNumber,
                  std::string outString, double fixedEpsilon = -999,
                  int64_t ulpsEpsilon = -999)
{
  // Check for equality and if not equal return difference
  double absoluteDiff;
  int64_t ulpsDiff;
  bool areEqual;

  if ((fixedEpsilon < 0) and (ulpsEpsilon < 0)) {
    areEqual = testingUtilities::nearlyEqualDbl(fiducialNumber, testNumber,
                                                absoluteDiff, ulpsDiff);
  } else if ((fixedEpsilon > 0) and (ulpsEpsilon < 0)) {
    areEqual = testingUtilities::nearlyEqualDbl(
        fiducialNumber, testNumber, absoluteDiff, ulpsDiff, fixedEpsilon);
  } else {
    areEqual = testingUtilities::nearlyEqualDbl(fiducialNumber, testNumber,
                                                absoluteDiff, ulpsDiff,
                                                fixedEpsilon, ulpsEpsilon);
  }

  std::stringstream outputMessage;
  outputMessage << std::setprecision(std::numeric_limits<double>::max_digits10)
                << "Difference in " << outString << std::endl
                << "The fiducial value is:       " << fiducialNumber
                << std::endl
                << "The test value is:           " << testNumber << std::endl
                << "The absolute difference is:  " << absoluteDiff << std::endl
                << "The ULP difference is:       " << ulpsDiff << std::endl;

  if (checkType == 0) {
    EXPECT_TRUE(areEqual) << outputMessage.str();
  } else if (checkType == 1) {
    ASSERT_TRUE(areEqual) << outputMessage.str();
  } else {
    throw std::runtime_error(
        "Incorrect template argument passed to "
        "checkResults. Options are 0 and 1 but " +
        std::to_string(checkType) + " was passed");
  }
}
// =========================================================================

// =========================================================================
/*!
 * \brief Holds a single std::string that's intended to be read only and
 * global. Use for storing the path of the root directory of Cholla
 *
 */
class GlobalString
{
 private:
  /// The path variable
  std::string _string;

 public:
  /*!
   * \brief Initializes the _path member variable. Should only be called
   * once in main
   *
   * \param inputPath The path to be store in _path
   */
  void init(std::string const &inputPath) { _string = inputPath; };

  /*!
   * \brief Get the String object
   *
   * \return std::string The string variable
   */
  std::string getString() { return _string; };
  GlobalString()  = default;
  ~GlobalString() = default;
};
// =========================================================================
}  // namespace testingUtilities

// Declare the global string variables so everything that imports this file
// has access to them
extern testingUtilities::GlobalString globalChollaRoot;
extern testingUtilities::GlobalString globalChollaBuild;
extern testingUtilities::GlobalString globalChollaMachine;
extern testingUtilities::GlobalString globalMpiLauncher;
extern bool globalRunCholla;
extern bool globalCompareSystemTestResults;
