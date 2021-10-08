/*!
 * \file systemTest.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Implementation file for the tools to run a system test
 *
 */

// STL includes
#include <stdlib.h>
#include <future>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <cstdio>

// External Libraries and Headers
#include <gtest/gtest.h>
#include <H5Cpp.h>

// Local includes
#include "../system_tests/system_tester.h" // Include the header file
#include "../utils/testing_utilities.h"

namespace systemTest
{
    namespace // Anonymous namespace with helper functions
    {
        // =====================================================================
        /*!
         * \brief Checks if the given file exists. Throws an exception if the
         * file does not exist.
         *
         * \param filePath The path to the file to check for
         */
        void checkFileExists(std::string const &filePath)
        {
            // TODO C++17 std::filesystem does this better
            std::fstream file;
            file.open(filePath);
            if (not file)
            {
                std::string errMessage = "Error: File '" + filePath + "' not found.";
                throw std::invalid_argument(errMessage);
            }
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Move a file. Throws an exception if the file does not exist.
         * or if the move was unsuccessful
         *
         * \param sourcePath The path the the file to be moved
         * \param destinationDirectory The path to the director the file should
         * be moved to
         */
        void safeMove(std::string const &sourcePath,
                      std::string const &destinationDirectory)
        {
            // TODO C++17 std::filesystem does this better
            checkFileExists(sourcePath);
            if(std::rename(sourcePath.c_str(), (destinationDirectory + "/" + sourcePath).c_str()) < 0)
            {
                std::string errMessage = "Error: File '"
                                         + sourcePath
                                         + "' could not be moved to '"
                                         + destinationDirectory
                                         + "`";
                throw std::invalid_argument(errMessage);
            }
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Load and HDF5 file, determine the number of time steps, and
         *        return the data sets
         *
         * \param[in]  filePath The path to the HDF5 file
         * \param[out] nSteps   The number of time steps in the simulation that
         *                      generated the data file
         * \param[out] h5Output The data from the HDF5 file
         */
        void loadHDF5(std::string const &filePath,
                      int &nSteps,
                      H5::H5File &h5Output)
        {
            // Open the file
            h5Output.openFile(filePath, H5F_ACC_RDONLY);

            // Get the time step attribute
            H5::Attribute tStepAttr = h5Output.openAttribute("n_step");
            tStepAttr.read(H5::PredType::NATIVE_INT, &nSteps);
        }
        // =====================================================================

        // =====================================================================
        /*!
         * \brief Generate all the paths that the system test needs and create
         * the output directory. All parameters are outputs
         *
         * \param[out] chollaPath The path to the Cholla executable
         * \param[out] chollaSettingsPath The path to the Cholla settings file
         * \param[out] fiducialDataFilePath The path to the fiducial data file
         * \param[out] outputDirectory The directory that will hold all the
         * result files. This function also creates this directory
         * \param[out] consoleOutputPath The path for where to put Cholla's
         * console output
         * \param[out] testDataFilePath The path to the test data the Cholla
         * will generate
         */
        void generatePathsAndCheckFiles(std::string &chollaPath,
                                        std::string &chollaSettingsPath,
                                        std::string &fiducialDataFilePath,
                                        std::string &outputDirectory,
                                        std::string &consoleOutputPath,
                                        std::string &testDataFilePath)
        {
            // Get the test name, with and underscore instead of a "." since
            // we're actually generating file names
            const testing::TestInfo* const test_info = testing::UnitTest::GetInstance()->current_test_info();
            std::stringstream nameStream;
            nameStream << test_info->test_suite_name() << "_" << test_info->name();
            std::string const fullTestName = nameStream.str();

            // Generate the input paths
            chollaPath           = globalChollaRoot.getString() + "/bin/cholla." + globalChollaBuild.getString() + "." + globalChollaMachine.getString();
            chollaSettingsPath   = globalChollaRoot.getString() + "/src/system_tests/input_files/" + fullTestName +".txt";
            fiducialDataFilePath = globalChollaRoot.getString() + "/src/system_tests/fiducial_data/" + fullTestName + ".h5";

            // Generate output paths, these files don't exist yet
            outputDirectory      = globalChollaRoot.getString() + "/bin/" + fullTestName;
            consoleOutputPath    = outputDirectory + "/" + fullTestName + "_console.log";
            testDataFilePath     = outputDirectory + "/1.h5.0";

            // Create the new directory and check that it exists
            // TODO: C++17: When we update to C++17 or newer this section should
            // TODO: use std::filesystem to create the directory and check that
            // TODO: it exists
            system(("mkdir -p -v " + outputDirectory).c_str());

            // Check that the files exist
            checkFileExists(chollaPath);
            checkFileExists(chollaSettingsPath);
            checkFileExists(fiducialDataFilePath);
        }
        // =====================================================================
    } // End Anonymous namespace

    // =========================================================================
    void systemTestRunner()
    {
        // Determine paths for everything and check that files exist
        std::string chollaPath,
                    chollaSettingsPath,
                    fiducialDataFilePath,
                    outputDirectory,
                    consoleOutputPath,
                    testDataFilePath;
        generatePathsAndCheckFiles(chollaPath,
                                   chollaSettingsPath,
                                   fiducialDataFilePath,
                                   outputDirectory,
                                   consoleOutputPath,
                                   testDataFilePath);

        // Launch Cholla asynchronously. Note that this dumps all console output
        // to the console log file as requested by the user. The parallel launch
        // might not actually speed things up at all. This is the first place to
        // look for performance improvements
        std::string const chollaRunCommand = chollaPath + " "
                                             + chollaSettingsPath + " "
                                             + "outdir=" + outputDirectory + "/"
                                             + " >> " + consoleOutputPath + " 2>&1 ";
        auto chollaProcess = std::async(std::launch::async, // Launch operation asynchronously rather than with lazy execution
                                        system,             // Choose the function to launch
                                        (chollaRunCommand).c_str()); // Args to send to "system" call

        // Load fiducial Data
        int fiducialNSteps;
        H5::H5File fiducialDataFile;
        loadHDF5(fiducialDataFilePath, fiducialNSteps, fiducialDataFile);

        // Wait for Cholla to Complete and copy output files to the test
        // directory
        chollaProcess.get();
        safeMove("run_output.log", outputDirectory);
        safeMove("run_timing.log", outputDirectory);

        // Load test data
        checkFileExists(testDataFilePath);
        int testNSteps;
        H5::H5File testDataFile;
        loadHDF5(testDataFilePath, testNSteps, testDataFile);

        // Perform Checks
        // ==============
        // Check that both files have the same number of data sets
        ASSERT_EQ(fiducialDataFile.getNumObjs(),
                  testDataFile.getNumObjs())
                  << "The two HDF5 files do not have the same number of datasets";

        // Check the number of time steps
        ASSERT_EQ(fiducialNSteps, testNSteps)
                  << "The number of time steps is not equal";

        // Loop through datasets and check each value
        for (size_t dataSetID = 0; dataSetID < fiducialDataFile.getNumObjs(); dataSetID++)
        {
            // Check that the dataset names are the same
            ASSERT_EQ(fiducialDataFile.getObjnameByIdx(dataSetID), testDataFile.getObjnameByIdx(dataSetID))
                      << "The data sets do not have the same name";

            // Assign DataSet objects
            H5::DataSet const fiducialDataSet = fiducialDataFile.openDataSet(fiducialDataFile.getObjnameByIdx(dataSetID));
            H5::DataSet const testDataSet     = testDataFile.openDataSet(testDataFile.getObjnameByIdx(dataSetID));

            // Determine dataset size/shape and check that it's correct
            H5::DataSpace fiducialDataSpace = fiducialDataSet.getSpace();
            H5::DataSpace testDataSpace     = testDataSet.getSpace();

            hsize_t fidDims[3] = {1,1,1}, testDims[3] = {1,1,1};
            ASSERT_TRUE(fiducialDataSpace.getSimpleExtentDims(fidDims) <= 3)
                        << "Expected 3 or fewer dimensions in dataset";
            ASSERT_TRUE(testDataSpace.getSimpleExtentDims(testDims) <= 3)
                        << "Expected 3 or fewer dimensions in dataset";
            ASSERT_EQ(fidDims[0], testDims[0])
                        << "Data sets are not the same length in the x-direction";
            ASSERT_EQ(fidDims[1], testDims[1])
                        << "Data sets are not the same length in the y-direction";
            ASSERT_EQ(fidDims[2], testDims[2])
                        << "Data sets are not the same length in the z-direction";

            // Allocate arrays, Note that I'm casting everything to double. Some
            // of the arrays are ints in the HDF5 file and if the casting
            // becomes an issue we can fix it later
            double *fiducialData = new double[fidDims[0] * fidDims[1] * fidDims[2]]();
            double *testData = new double[testDims[0] * testDims[1] * testDims[2]]();

            // Read in data
            fiducialDataSet.read(fiducialData, H5::PredType::NATIVE_DOUBLE);
            testDataSet.read(testData, H5::PredType::NATIVE_DOUBLE);

            // Compare values
            for (size_t i = 0; i < fidDims[0]; i++)
            {
                for (size_t j = 0; j < fidDims[1]; j++)
                {
                    for (size_t k = 0; k < fidDims[2]; k++)
                    {
                        size_t index = (i * fidDims[1] + j) * fidDims[2] + k;

                        // Check for equality and iff not equal return difference
                        double absoluteDiff;
                        int64_t ulpsDiff;
                        bool areEqual = testingUtilities::nearlyEqualDbl(fiducialData[index],
                                                                         testData[index],
                                                                         absoluteDiff,
                                                                         ulpsDiff);
                        ASSERT_TRUE(areEqual)
                            << std::endl
                            << "Difference in "
                            << fiducialDataFile.getObjnameByIdx(dataSetID)
                            << " dataset at ["
                            << i << "," << j << "," << k <<"]" << std::endl
                            << "The fiducial value is:       " << fiducialData[index] << std::endl
                            << "The test value is:           " << testData[index]     << std::endl
                            << "The absolute difference is:  " << absoluteDiff        << std::endl
                            << "The ULP difference is:       " << ulpsDiff            << std::endl;
                    }

                }
            }
            delete[] fiducialData;
            delete[] testData;
        } // End Dataset loop
    }
    // =========================================================================
} // namespace systemTest
