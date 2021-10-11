/*!
 * \file systemTest.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Implementation file for the tools to run a system test
 *
 */

// STL includes
#include <stdlib.h>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <algorithm>
#include <cmath>

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../system_tests/system_tester.h" // Include the header file
#include "../utils/testing_utilities.h"

// =============================================================================
// Public Members
// =============================================================================

// =============================================================================
void systemTest::SystemTestRunner::runTest()
{
    // Launch Cholla. Note that this dumps all console output to the console log
    // file as requested by the user. The parallel launch might not actually
    // speed things up at all. This is the first place to look for performance
    // improvements
    std::string const chollaRunCommand = _chollaPath + " "
                                         + _chollaSettingsPath + " "
                                         + _chollaLaunchParams + " "
                                         + "outdir=" + _outputDirectory + "/"
                                         + " >> " + _consoleOutputPath + " 2>&1 ";
    system((chollaRunCommand).c_str()); // Args to send to "system" call
    _safeMove("run_output.log", _outputDirectory);
    _safeMove("run_timing.log", _outputDirectory);

    // Make sure we have all the required data files and open the test data file
    _checkFileExists(_testFilePath);
    _testFile.openFile(_testFilePath, H5F_ACC_RDONLY);
    _testDataSetNames = _findDataSetNames(_testFile);

    // Start Performing Checks
    // =======================
    // Check the number of time steps
    if (_compareNumTimeSteps) _checkNumTimeSteps();

    // Check that the test file has as many, or more, datasets than the fiducial
    // file. Provide a warning if the datasets are not the same size
    EXPECT_GE(_testDataSetNames.size(), _fiducialDataSetNames.size());
    if (_testDataSetNames.size() != _fiducialDataSetNames.size())
    {
        std::cout << std::endl << "Warning: The test data has "
        << _testDataSetNames.size() <<  " datasets and the fiducial data has "
        << _fiducialDataSetNames.size() << " datasets" << std::endl << std::endl;
    }

    // Loop over the datasets to be tested
    for (auto dataSetName: _fiducialDataSetNames)
    {
        // check that the test data has the dataset in it
        ASSERT_EQ(std::count(_testDataSetNames.begin(), _testDataSetNames.end(), dataSetName), 1)
            << "The test data does not contain the dataset '" + dataSetName
            + "' or contains it more than once.";

        // Get test data array
        H5::DataSet const testDataSet = _testFile.openDataSet(dataSetName);
        H5::DataSpace testDataSpace   = testDataSet.getSpace();

        hsize_t testDims[3] = {1,1,1};
        ASSERT_LE(testDataSpace.getSimpleExtentDims(testDims), 3)
                << "Expected 3 or fewer dimensions in dataset";

        // Allocate arrays, Note that I'm casting everything to double. Some
        // of the arrays are ints in the HDF5 file and if the casting
        // becomes an issue we can fix it later
        size_t const testSize = testDims[0] * testDims[1] * testDims[2];
        auto testData = std::shared_ptr<double[]>{ new double[testSize] };
        testDataSet.read(testData.get(), H5::PredType::NATIVE_DOUBLE);

        // Get fiducial data array
        size_t fiducialSize;
        std::shared_ptr<double[]> fiducialData = _getFiducialArray(dataSetName,
                                                                   fiducialSize);

        // Check that they're the same length
        ASSERT_EQ(fiducialSize, testSize) << "The fiducial and test '"
                                          << dataSetName
                                          << "' datasets are not the same length";

        // Compare values
        for (size_t i = 0; i < testDims[0]; i++)
        {
            for (size_t j = 0; j < testDims[1]; j++)
            {
                for (size_t k = 0; k < testDims[2]; k++)
                {
                    size_t index = (i * testDims[1] + j) * testDims[2] + k;

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
                        << dataSetName
                        << " dataset at ["
                        << i << "," << j << "," << k <<"]" << std::endl
                        << "The fiducial value is:       " << fiducialData[index] << std::endl
                        << "The test value is:           " << testData[index]     << std::endl
                        << "The absolute difference is:  " << absoluteDiff        << std::endl
                        << "The ULP difference is:       " << ulpsDiff            << std::endl;
                }
            }
        }
    }
}
// =============================================================================

// =============================================================================
void systemTest::SystemTestRunner::setFiducialData(std::string const &fieldName,
                                                   std::shared_ptr<double[]> const &dataArr,
                                                   size_t const &sizeOfArr,
                                                   int const &numTimeSteps)
{
    // First check if there's a fiducial data file
    if (_fiducialFileExists)
    {
        std::string errMessage = "Error: Fiducial data file already exists for test '"
                                 + _fullTestFileName
                                 + "' and cannot be overwritten.";
        throw std::runtime_error(errMessage);
    }

    // Check if we need to set the number of time steps
    if (numTimeSteps >= 0) _numFiducialTimeSteps = numTimeSteps;

    // Put new array into map and its size
    _fiducialDataSets[fieldName] = dataArr;
    _fiducialArrSize[fieldName] = sizeOfArr;
}
// =============================================================================

// =============================================================================
std::shared_ptr<double[]> systemTest::SystemTestRunner::generateConstantArray(
                                                        double const &value,
                                                        size_t const &nx,
                                                        size_t const &ny,
                                                        size_t const &nz)
{
    size_t const arrSize = nx*ny*nz;
    auto outArray = std::shared_ptr<double[]>{ new double[arrSize] };
    for (size_t i = 0; i < arrSize; i++)
    {
        outArray[i] = value;
    }
    return outArray;
}
// =============================================================================

// =============================================================================
std::shared_ptr<double[]> systemTest::SystemTestRunner::generateSineArray(
                                                        double const &offset,
                                                        double const &amplitude,
                                                        double const &kx,
                                                        double const &ky,
                                                        double const &kz,
                                                        double const &phase,
                                                        size_t const &nx,
                                                        size_t const &ny,
                                                        size_t const &nz)
{
    size_t const arrSize = nx*ny*nz;
    auto outArray = std::shared_ptr<double[]>{ new double[arrSize] };
    for (size_t i = 0; i < nx; i++)
    {
        for (size_t j = 0; j < ny; j++)
        {
            for (size_t k = 0; k < nz; k++)
            {
                double value    = offset + amplitude
                                  * std::sin(kx*i + ky*j + kz*k + phase);

                size_t index    = (i * ny + j) * nz + k;
                outArray[index] = value;
            }
        }
    }
    return outArray;
}
// =============================================================================

// =============================================================================
// Constructor
systemTest::SystemTestRunner::SystemTestRunner(bool const &useFiducialFile,
                                               bool const &useSettingsFile)
{
    // Get the test name, with and underscore instead of a "." since
    // we're actually generating file names
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream nameStream;
    nameStream << test_info->test_suite_name() << "_" << test_info->name();
    _fullTestFileName = nameStream.str();

    // Generate the input paths
    _chollaPath         = ::globalChollaRoot.getString()
                          + "/bin/cholla."
                          + ::globalChollaBuild.getString()
                          + "." + ::globalChollaMachine.getString();
    _chollaSettingsPath = ::globalChollaRoot.getString()
                          + "/src/system_tests/input_files/"
                          + _fullTestFileName +".txt";
    _fiducialFilePath   = ::globalChollaRoot.getString()
                          + "/src/system_tests/fiducial_data/"
                          + _fullTestFileName + ".h5";

    // Generate output paths, these files don't exist yet
    _outputDirectory    = ::globalChollaRoot.getString() + "/bin/" + _fullTestFileName;
    _consoleOutputPath  = _outputDirectory + "/" + _fullTestFileName + "_console.log";
    _testFilePath       = _outputDirectory + "/1.h5.0";

    // Create the new directory and check that it exists
    // TODO: C++17: When we update to C++17 or newer this section should
    // TODO: use std::filesystem to create the directory and check that
    // TODO: it exists
    if (mkdir(_outputDirectory.c_str(), 0777) == -1)
    {
        std::string errMessage = "Error: Directory '"
                                 + _outputDirectory
                                 + "' could not be created.";
        throw std::runtime_error(errMessage);
    }

    // Check that the files exist and load fiducial HDF5 file if required
    _checkFileExists(_chollaPath);
    if (useSettingsFile) _checkFileExists(_chollaSettingsPath);
    if (useFiducialFile)
    {
        _checkFileExists(_fiducialFilePath);
        _fiducialFile.openFile(_fiducialFilePath, H5F_ACC_RDONLY);
        _fiducialDataSetNames = _findDataSetNames(_fiducialFile);
        _fiducialFileExists   = true;
    };
}
// =============================================================================

// =============================================================================
// Destructor
systemTest::SystemTestRunner::~SystemTestRunner()
{
    _fiducialFile.close();
    _testFile.close();
}
// =============================================================================

// =============================================================================
// Private Members
// =============================================================================

// =============================================================================
void systemTest::SystemTestRunner::_checkFileExists(std::string const &filePath)
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
// =============================================================================

// =============================================================================
void systemTest::SystemTestRunner::_safeMove(std::string const &sourcePath,
                                             std::string const &destinationDirectory)
{
    // TODO C++17 std::filesystem does this better
    _checkFileExists(sourcePath);
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
// =============================================================================

// =============================================================================
void systemTest::SystemTestRunner::_checkNumTimeSteps()
{
    int fiducialNSteps, testNSteps;

    H5::Attribute tStepAttr = _testFile.openAttribute("n_step");
    tStepAttr.read(H5::PredType::NATIVE_INT, &testNSteps);

    if (_fiducialFileExists)
    {
        tStepAttr = _fiducialFile.openAttribute("n_step");
        tStepAttr.read(H5::PredType::NATIVE_INT, &fiducialNSteps);
    }
    else
    {
        fiducialNSteps = _numFiducialTimeSteps;
    }

    EXPECT_EQ(fiducialNSteps, testNSteps)
              << "The number of time steps is not equal";
};
// =============================================================================

// =============================================================================
std::shared_ptr<double[]> systemTest::SystemTestRunner::_getFiducialArray(
        std::string const &dataSetName,
        size_t &length)
{
    if (_fiducialFileExists)
    {
        // Read in fiducial Data
        H5::DataSet const fiducialDataSet = _fiducialFile.openDataSet(dataSetName);

        // Determine dataset size/shape and check that it's correct
        H5::DataSpace fiducialDataSpace = fiducialDataSet.getSpace();

        hsize_t fidDims[3] = {1,1,1};
        fiducialDataSpace.getSimpleExtentDims(fidDims);

        // Allocate arrays, Note that I'm casting everything to double. Some
        // of the arrays are ints in the HDF5 file and if the casting
        // becomes an issue we can fix it later
        length = fidDims[0] * fidDims[1] * fidDims[2];
        auto fiducialData = std::shared_ptr<double[]>{ new double[length] };

        // Read in data
        fiducialDataSet.read(fiducialData.get(), H5::PredType::NATIVE_DOUBLE);
        return fiducialData;
    }
    else
    {
        length = _fiducialArrSize[dataSetName];
        return _fiducialDataSets[dataSetName];
    }
}
// =============================================================================

// =============================================================================
std::vector<std::string> systemTest::SystemTestRunner::_findDataSetNames(
                                                    H5::H5File const &inputFile)
{
    std::vector<std::string> outputVector;

    for (size_t dataSetID = 0;
         dataSetID < inputFile.getNumObjs();
         dataSetID++)
    {
        outputVector.push_back(inputFile.getObjnameByIdx(dataSetID));
    }
    return outputVector;
};
// =============================================================================