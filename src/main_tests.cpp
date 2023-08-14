/*!
 * \file main_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the main function for tests
 *
 */

// STL includes
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "utils/testing_utilities.h"

/// This is the global variable to store the path to the root of Cholla
testingUtilities::GlobalString globalChollaRoot;
testingUtilities::GlobalString globalChollaBuild;
testingUtilities::GlobalString globalChollaMachine;
testingUtilities::GlobalString globalMpiLauncher;
bool globalRunCholla;
bool globalCompareSystemTestResults;

/*!
 * \brief Class for parsing input flags. Modified from
 * https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
 *
 */
class InputParser
{
 public:
  // =====================================================================
  /*!
   * \brief Get the option that follows the given flag. Also checks that
   * the flag exists and is not empty
   *
   * \param option The string option to look for
   * \return const std::string& The option the follows a given flag
   */
  const std::string &getCmdOption(const std::string &option) const
  {
    // First check that the option exists
    if (not cmdOptionExists(option)) {
      std::string errMessage = "Error: argument '" + option + "' not found. ";
      throw std::invalid_argument(errMessage);
    }

    std::vector<std::string>::const_iterator itr;
    itr = std::find(this->_tokens.begin(), this->_tokens.end(), option);
    if (itr != this->_tokens.end() && ++itr != this->_tokens.end()) {
      return *itr;
    } else {
      std::string errMessage = "Error: empty argument '" + option + "'";
      throw std::invalid_argument(errMessage);
    }
  }
  // =====================================================================

  // =====================================================================
  /*!
   * \brief Checks that an option exists. Returns True if it exists and
   * False otherwise
   *
   * \param option The option flag to search for
   * \return true The option flag exists in argv
   * \return false The option flage does not exist in argv
   */
  bool cmdOptionExists(const std::string &option) const
  {
    return std::find(this->_tokens.begin(), this->_tokens.end(), option) != this->_tokens.end();
  }
  // =====================================================================

  // =====================================================================
  // constructor and destructor
  /*!
   * \brief Construct a new Input Parser object
   *
   * \param argc argc from main
   * \param argv argv from main
   */
  InputParser(int &argc, char **argv)
  {
    for (int i = 1; i < argc; ++i) {
      this->_tokens.emplace_back(argv[i]);
    }
  }
  ~InputParser() = default;
  // =====================================================================
 private:
  std::vector<std::string> _tokens;
};

/*!
 * \brief The main function for testing
 *
 * \param argc The number of CLI arguments
 * \param argv A list of the CLI arguments. Requires that `--cholla-root` be set
 * to the root directory of Cholla, `--build-type` be set to the make type used
 * to build cholla and the tests, and `--machine` be set to the name of the
 * machine being used, this must be the same name as in the `make.host` file
 * \return int
 */
int main(int argc, char **argv)
{
  // First we initialize Googletest. Note, this removes all gtest related
  // arguments from argv and argc
  ::testing::InitGoogleTest(&argc, argv);

  // Make sure death tests are threadsafe. This is potentially much slower than
  // using "fast" instead of "threadsafe" but it makes sure tests are threadsafe
  // in a multithreaded environment. If the performance becomes an issue we can
  // try "fast", it can also be set on a test by test basis
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";

  // Initialize global variables
  InputParser input(argc, argv);
  globalChollaRoot.init(input.getCmdOption("--cholla-root"));
  globalChollaBuild.init(input.getCmdOption("--build-type"));
  globalChollaMachine.init(input.getCmdOption("--machine"));
  if (input.cmdOptionExists("--mpi-launcher")) {
    globalMpiLauncher.init(input.getCmdOption("--mpi-launcher"));
  } else {
    globalMpiLauncher.init("mpirun -np");
  }

  globalRunCholla                = not input.cmdOptionExists("--runCholla=false");
  globalCompareSystemTestResults = not input.cmdOptionExists("--compareSystemTestResults=false");

  // Run test and return result
  return RUN_ALL_TESTS();
}
