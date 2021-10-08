/*!
 * \file main_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the main function for tests
 *
 */

// STL includes
#include <string>
#include <algorithm>
#include <vector>
#include <stdexcept>

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "utils/testing_utilities.h"

/// This is the global variable to store the path to the root of Cholla
testingUtilities::GlobalString globalChollaRoot;
testingUtilities::GlobalString globalChollaBuild;
testingUtilities::GlobalString globalChollaMachine;


/*!
 * \brief Class for parsing input flags. Modified from
 * https://stackoverflow.com/questions/865668/parsing-command-line-arguments-in-c
 *
 */
class InputParser{
    public:
        // =====================================================================
        /*!
         * \brief Get the option that follows the given flag. Also checks that
         * the flag exists and is not empty
         *
         * \param option The string option to look for
         * \return const std::string& The option the follows a given flag
         */
        const std::string& getCmdOption(const std::string &option) const
        {
            // First check that the option exists
            if(not cmdOptionExists(option))
            {
                std::string errMessage = "Error: argument '" + option + "' not found. ";
                throw std::invalid_argument(errMessage);
            }

            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end())
            {
                return *itr;
            }
            else
            {
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
            return std::find(this->tokens.begin(), this->tokens.end(), option)
            != this->tokens.end();
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
        InputParser (int &argc, char **argv)
        {
            for (int i=1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
        }
        ~InputParser() = default;
        // =====================================================================
    private:
        std::vector <std::string> tokens;
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
    // TODO: Reenable this whenever I can get it to compile properly. In the
    // TODO: meantime use the command line argument
    // TODO: --gtest_death_test_style=(fast|threadsafe)
    // GTEST_FLAG_SET(death_test_style, "threadsafe");
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
    // Initialize the global cholla root path variable from CLI arguments
    InputParser input(argc, argv);
    globalChollaRoot.initPath(input.getCmdOption("--cholla-root"));
    globalChollaBuild.initPath(input.getCmdOption("--build-type"));
    globalChollaMachine.initPath(input.getCmdOption("--machine"));

    // Only run those tests that correspond to the current make type
    std::string upperBuildType = globalChollaBuild.getString();
    std::transform(upperBuildType.begin(),
                   upperBuildType.end(),
                   upperBuildType.begin(),
                   ::toupper);
    ::testing::GTEST_FLAG(filter) = "*t"+ upperBuildType + "*:*tALL*";

    // Run test and return result
    return RUN_ALL_TESTS();
}
