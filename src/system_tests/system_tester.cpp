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
#include <numeric>

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
    /// Only run if this variable is set to `true`. Generally this and
    /// globalCompareSystemTestResults should only be used for large MPI / tests
    /// where the user wishes to separate the execution of cholla and the /
    /// comparison of results onto different machines/jobs
    if (globalRunCholla)
    {
        // Launch Cholla. Note that this dumps all console output to the console
        // log file as requested by the user. The parallel launch might not
        // actually speed things up at all. This is the first place to look for
        // performance improvements
        std::string const chollaRunCommand = globalMpiLauncher.getString() + " "
                                            + std::to_string(numMpiRanks) + " "
                                            + _chollaPath + " "
                                            + _chollaSettingsPath + " "
                                            + _chollaLaunchParams + " "
                                            + "outdir=" + _outputDirectory + "/"
                                            + " >> " + _consoleOutputPath + " 2>&1 ";
        system(("echo Launch Command: " + chollaRunCommand + " >> " + _consoleOutputPath).c_str());
        system((chollaRunCommand).c_str()); // Args to send to "system" call
        _safeMove("run_output.log", _outputDirectory);
        _safeMove("run_timing.log", _outputDirectory);
    }

    /// If set to false then no comparison will be performed. Generally this and
    /// globalRunCholla should only be used for large MPI tests where the user
    /// wishes to separate the execution of cholla and the comparison of results
    /// onto different machines/jobs
    if (not globalCompareSystemTestResults) return;

    // Make sure we have all the required data files and open the test data file
    _testHydroFieldsFileVec.resize(numMpiRanks);
    _testParticlesFileVec.resize(numMpiRanks);
    for (size_t fileIndex = 0; fileIndex < numMpiRanks; fileIndex++)
    {
        // Load the hydro data
        if (_hydroDataExists)
        {
            std::string fileName = "/1.h5." + std::to_string(fileIndex);
            _checkFileExists(_outputDirectory + fileName);
            _testHydroFieldsFileVec[fileIndex].openFile(_outputDirectory + fileName,
                                                        H5F_ACC_RDONLY);
        }

        // Load the particles data
        if (_particleDataExists)
        {
            std::string fileName = "/1_particles.h5." + std::to_string(fileIndex);
            _checkFileExists(_outputDirectory + fileName);
            _testParticlesFileVec[fileIndex].openFile(_outputDirectory + fileName,
                                                      H5F_ACC_RDONLY);
        }
    }

    // If this is a particle build then read in the IDs and generate the sorting
    // vector
    if (_particleDataExists)
    {
        _testParticleIDs     = _loadTestParticleData("particle_IDs");

        _fiducialParticleIDs = _loadFiducialParticleData("particle_IDs");
    }

    // Get the list of test dataset names
    if (_hydroDataExists)
        _testDataSetNames = _findDataSetNames(_testHydroFieldsFileVec[0]);
    if (_particleDataExists)
    {
        // Load the data, replace the density value with the new name, then append
        std::vector<std::string> particleNames = _findDataSetNames(_testParticlesFileVec[0]);
        auto iter = std::find(particleNames.begin(), particleNames.end(), "density");
        *iter = "particle_density";

        _testDataSetNames.insert(_testDataSetNames.end(),
                                 particleNames.begin(),
                                 particleNames.end());
    }

    // Start Performing Checks
    // =======================
    // Check the number of time steps
    if (_compareNumTimeSteps) _checkNumTimeSteps();

    // Check that the test file has as many, or more, datasets than the fiducial
    // file. Provide a warning if the datasets are not the same size
    EXPECT_GE(_testDataSetNames.size(), _fiducialDataSetNames.size())
        << std::endl
        << "Warning: The test data has "
        << _testDataSetNames.size()
        <<  " datasets and the fiducial data has "
        << _fiducialDataSetNames.size()
        << " datasets" << std::endl << std::endl;

    // Loop over the datasets to be tested
    for (auto dataSetName: _fiducialDataSetNames)
    {
        // check that the test data has the dataset in it
        ASSERT_EQ(std::count(_testDataSetNames.begin(), _testDataSetNames.end(), dataSetName), 1)
            << "The test data does not contain the dataset '" + dataSetName
            + "' or contains it more than once.";

        // Get data vectors
        std::vector<size_t> testDims(3,1);
        std::vector<double> testData;
        std::vector<double> fiducialData;
        // This is just a vector of all the different dataset names for
        // particles to help choose whether to call _loadTestParticleData
        // or _loadTestFieldData
        std::vector<std::string> particleIDs = {"particle_IDs",
                                                "pos_x",
                                                "pos_y",
                                                "pos_z",
                                                "vel_x",
                                                "vel_y",
                                                "vel_z"};
        if (std::find(particleIDs.begin(), particleIDs.end(), dataSetName)
            != particleIDs.end())
        {
            // This is a particle data set

            // Set some basic parameters
            testDims[0] = _testTotalNumParticles;

            // Load in the data. Note the special handling for particle_IDs
            if (dataSetName == "particle_IDs")
            {
                testData     = _testParticleIDs;
                fiducialData = _fiducialParticleIDs;
            }
            else
            {
                testData     = _loadTestParticleData(dataSetName);
                fiducialData = _loadFiducialParticleData(dataSetName);
            }
        }
        else
        {
            // This is a field data set
            testData = _loadTestFieldData(dataSetName,
                                          testDims);
            // Get fiducial data
            fiducialData = _loadFiducialFieldData(dataSetName);
        }

        // Check that they're the same length
        ASSERT_EQ(fiducialData.size(), testData.size())
                                        << "The fiducial and test '"
                                        << dataSetName
                                        << "' datasets are not the same length";

        // Compare values
        for (size_t i = 0; i < testDims[0]; i++)
        {
            for (size_t j = 0; j < testDims[1]; j++)
            {
                for (size_t k = 0; k < testDims[2]; k++)
                {
                    size_t index = (i * testDims[1] * testDims[2]) + (j * testDims[2]) + k;

                    // Check for equality and iff not equal return difference
                    double absoluteDiff;
                    int64_t ulpsDiff;
                    // Fixed epsilon is changed from the default since AMD/Clang
                    // appear to differ from NVIDIA/GCC/XL by roughly 1E-12
                    double fixedEpsilon = 5.0E-12;
                    bool areEqual = testingUtilities::nearlyEqualDbl(fiducialData.at(index),
                                                                     testData.at(index),
                                                                     absoluteDiff,
                                                                     ulpsDiff,
                                                                     fixedEpsilon);
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
                                                   std::vector<double> const &dataVec)
{
    // First check if there's a fiducial data file
    if (_fiducialFileExists)
    {
        std::string errMessage = "Error: Fiducial data file already exists for test '"
                                 + _fullTestFileName
                                 + "' and cannot be overwritten.";
        throw std::runtime_error(errMessage);
    }

    // Put new vector into map
    _fiducialDataSets[fieldName] = dataVec;
}
// =============================================================================

// =============================================================================
std::vector<double> systemTest::SystemTestRunner::generateConstantData(
                                                        double const &value,
                                                        size_t const &nx,
                                                        size_t const &ny,
                                                        size_t const &nz)
{
    size_t const length = nx*ny*nz;
    std::vector<double> outVec(length);
    for (size_t i = 0; i < length; i++)
    {
        outVec[i] = value;
    }
    return outVec;
}
// =============================================================================

// =============================================================================
std::vector<double> systemTest::SystemTestRunner::generateSineData(
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
    size_t const length = nx*ny*nz;
    std::vector<double> outVec(length);
    for (size_t i = 0; i < nx; i++)
    {
        for (size_t j = 0; j < ny; j++)
        {
            for (size_t k = 0; k < nz; k++)
            {
                double value = offset + amplitude
                               * std::sin(kx*i + ky*j + kz*k + phase);

                size_t index = (i * ny * nz) + (j * nz) + k;
                outVec[index] = value;
            }
        }
    }
    return outVec;
}
// =============================================================================

// =============================================================================
// Constructor
systemTest::SystemTestRunner::SystemTestRunner(bool const &particleData,
                                               bool const &hydroData,
                                               bool const &useFiducialFile,
                                               bool const &useSettingsFile)
    :
    _particleDataExists(particleData),
    _hydroDataExists(hydroData)
{
    // Get the test name, with and underscore instead of a "." since
    // we're actually generating file names
    const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::stringstream nameStream;
    std::string suiteName = test_info->test_suite_name();
    suiteName = suiteName.substr(suiteName.find("/")+1, suiteName.length());
    nameStream << suiteName << "_" << test_info->name();
    std::string fullTestName = nameStream.str();
    _fullTestFileName = fullTestName.substr(0, fullTestName.find("/"));

    // Generate the input paths. Strip out everything after a "/" since that
    // probably indicates a parameterized test
    _chollaPath         = ::globalChollaRoot.getString()
                          + "/bin/cholla."
                          + ::globalChollaBuild.getString()
                          + "." + ::globalChollaMachine.getString();
    _chollaSettingsPath = ::globalChollaRoot.getString()
                          + "/src/system_tests/input_files/"
                          + _fullTestFileName + ".txt";
    _fiducialFilePath   = ::globalChollaRoot.getString()
                          + "/src/system_tests/fiducial_data/"
                          + _fullTestFileName + ".h5";

    // Generate output paths, these files don't exist yet
    _outputDirectory    = ::globalChollaRoot.getString() + "/bin/" + fullTestName;
    _consoleOutputPath  = _outputDirectory + "/" + _fullTestFileName + "_console.log";

    // Create the new directory and check that it exists
    // TODO: C++17: When we update to C++17 or newer this section should
    // TODO: use std::filesystem to create the directory and check that
    // TODO: it exists
    if (system(("mkdir --parents " + _outputDirectory).c_str()) != 0)
    {
        std::cerr << "Warning: Directory '"
                      + _outputDirectory
                      + "' either already exists or could not be created."
                      << std::endl;
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
    for (size_t i = 0; i < _testHydroFieldsFileVec.size(); i++)
    {
        if (_hydroDataExists) _testHydroFieldsFileVec[i].close();
        if (_particleDataExists) _testParticlesFileVec[i].close();
    }
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

    H5::Attribute tStepAttr;
    if (_hydroDataExists)
    {
        tStepAttr = _testHydroFieldsFileVec[0].openAttribute("n_step");
    }
    else if (_particleDataExists)
    {
        tStepAttr = _testParticlesFileVec[0].openAttribute("n_step");
    }
    else
    {
        std::string errMessage = "Error: Both hydro and particle data are  turned off.";
        throw std::invalid_argument(errMessage);
    }

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
std::vector<double> systemTest::SystemTestRunner::_loadTestFieldData(
        std::string dataSetName,
        std::vector<size_t> &testDims)
{
    // Get the file we're using
    std::vector<H5::H5File> file;
    if (dataSetName == "particle_density")
    {
        file = _testParticlesFileVec;
        dataSetName = "density";
    }
    else
    {
        file = _testHydroFieldsFileVec;
    }

    // Get the size of each dimension
    H5::Attribute dimensions = file[0].openAttribute("dims");
    dimensions.read(H5::PredType::NATIVE_ULONG, testDims.data());

    // Allocate the vector
    std::vector<double> testData(testDims[0] * testDims[1] * testDims[2]);

    for (size_t rank = 0; rank < numMpiRanks; rank++)
    {
        // Open the dataset
        H5::DataSet const testDataSet = file[rank].openDataSet(dataSetName);

        // Determine dataset size/shape and check that it's correct
        H5::DataSpace const testDataSpace = testDataSet.getSpace();

        std::vector<hsize_t> tempDims{1,1,1};
        int numTestDims = testDataSpace.getSimpleExtentDims(tempDims.data());

        // Allocate vectors, Note that I'm casting everything to double. Some
        // of the vectors are ints in the HDF5 file and if the casting
        // becomes an issue we can fix it later
        std::vector<double> tempArr(tempDims[0] * tempDims[1] * tempDims[2]);

        // Read in data
        testDataSet.read(tempArr.data(), H5::PredType::NATIVE_DOUBLE);

        // Get offset
        std::vector<int> offset(3,1);
        H5::Attribute offsetAttr = file[rank].openAttribute("offset");
        offsetAttr.read(H5::PredType::NATIVE_INT, offset.data());

        // Get dims_local
        std::vector<int> dimsLocal(3,1);
        H5::Attribute dimsLocalAttr = file[rank].openAttribute("dims_local");
        dimsLocalAttr.read(H5::PredType::NATIVE_INT, dimsLocal.data());

        // Now we add the data to the larger vector
        size_t localIndex = 0;
        for (size_t i = offset[0]; i < offset[0] + dimsLocal[0]; i++)
        {
            for (size_t j = offset[1]; j < offset[1] + dimsLocal[1]; j++)
            {
                for (size_t k = offset[2]; k < offset[2] + dimsLocal[2]; k++)
                {
                    // Compute the location to put the next element
                    size_t overallIndex = (i * testDims[1] * testDims[2]) + (j * testDims[2]) + k;

                    // Perform copy
                    testData[overallIndex] = tempArr[localIndex];

                    // Increment local index
                    localIndex++;
                }
            }
        }
    }

    // Return the entire, concatenated, dataset
    return testData;
}
// =============================================================================

// =============================================================================
std::vector<double> systemTest::SystemTestRunner::_loadTestParticleData(
        std::string const &dataSetName)
{
    // Determine the total number of particles
    if (_testTotalNumParticles == 0)
    {
        for (auto file: _testParticlesFileVec)
        {
            // Open the dataset
            H5::DataSet const dataSet = file.openDataSet(dataSetName);

            // Determine dataset size/shape and check that it's correct
            H5::DataSpace dataSpace = dataSet.getSpace();

            // Get the number of elements and increase the total count
            size_t localNumParticles = dataSpace.getSimpleExtentNpoints();
            _testTotalNumParticles += localNumParticles;
        }
    }

    // Allocate the vectors
    std::vector<double> unsortedTestData;
    std::vector<double> testData(_testTotalNumParticles);

    // Load in the data
    for (size_t rank = 0; rank < numMpiRanks; rank++)
    {
        // Open the dataset
        H5::DataSet const testDataSet = _testParticlesFileVec[rank].openDataSet(dataSetName);

        // Determine dataset size/shape and check that it's correct
        H5::DataSpace const testDataSpace = testDataSet.getSpace();

        size_t localNumParticles = testDataSpace.getSimpleExtentNpoints();
        std::vector<double> tempVector(localNumParticles);

        // Read in data
        testDataSet.read(tempVector.data(),
                         H5::PredType::NATIVE_DOUBLE);
        unsortedTestData.insert(unsortedTestData.end(),
                                tempVector.begin(),
                                tempVector.end() );
    }

    // Generate the sorting vector if it's not already generated
    std::vector<size_t> tempSortedIndices;
    if (dataSetName == "particle_IDs")
    {
        tempSortedIndices.resize(_testTotalNumParticles);
        std::iota(tempSortedIndices.begin(), tempSortedIndices.end(), 0);
        std::sort(tempSortedIndices.begin(), tempSortedIndices.end(),
                [&](size_t A, size_t B) -> bool {
                        return unsortedTestData[A] < unsortedTestData[B];
                    });
    }
    std::vector<size_t> static const sortedIndices = tempSortedIndices;

    // Sort the vector
    for (size_t i = 0; i < _testTotalNumParticles; i++)
    {
        testData.at(i) = unsortedTestData.at(sortedIndices.at(i));
    }

    // Return the entire dataset fully concatenated and sorted
    return testData;
}
// =============================================================================

// =============================================================================
std::vector<double> systemTest::SystemTestRunner::_loadFiducialFieldData(
    std::string const &dataSetName)
{
    if (_fiducialFileExists)
    {
        // Open the dataset
        H5::DataSet const fiducialDataSet = _fiducialFile.openDataSet(dataSetName);

        // Determine dataset size/shape and check that it's correct
        H5::DataSpace fiducialDataSpace = fiducialDataSet.getSpace();

        std::vector<hsize_t> fidDims{1,1,1};
        fiducialDataSpace.getSimpleExtentDims(fidDims.data());

        // Allocate vectors, Note that I'm casting everything to double. Some
        // of the vectors are ints in the HDF5 file and if the casting
        // becomes an issue we can fix it later
        std::vector<double> fiducialData(fidDims[0] * fidDims[1] * fidDims[2]);

        // Read in data
        fiducialDataSet.read(fiducialData.data(), H5::PredType::NATIVE_DOUBLE);
        return fiducialData;
    }
    else
    {
        return _fiducialDataSets[dataSetName];
    }
}
// =============================================================================

// =============================================================================
std::vector<double> systemTest::SystemTestRunner::_loadFiducialParticleData(
        std::string const &dataSetName)
{
    // Determine the total number of particles
    if (_fiducialTotalNumParticles == 0)
    {
        // Open the dataset
        H5::DataSet const dataSet = _fiducialFile.openDataSet(dataSetName);

        // Determine dataset size/shape and check that it's correct
        H5::DataSpace dataSpace = dataSet.getSpace();

        // Get the number of elements and increase the total count
        size_t localNumParticles = dataSpace.getSimpleExtentNpoints();
        _fiducialTotalNumParticles += localNumParticles;
    }

    // Allocate the vectors
    std::vector<double> unsortedFiducialData(_fiducialTotalNumParticles);
    std::vector<double> fiducialData(_fiducialTotalNumParticles);

    // Load in the data
    // Open the dataset
    H5::DataSet const fiducialDataSet = _fiducialFile.openDataSet(dataSetName);

    // Determine dataset size/shape and check that it's correct
    H5::DataSpace const testDataSpace = fiducialDataSet.getSpace();

    size_t localNumParticles = testDataSpace.getSimpleExtentNpoints();

    // Read in data
    fiducialDataSet.read(unsortedFiducialData.data(),
                         H5::PredType::NATIVE_DOUBLE);

    // Generate the sorting vector if it's not already generated
    std::vector<size_t> tempSortedIndices;
    if (dataSetName == "particle_IDs")
    {
        tempSortedIndices.resize(_fiducialTotalNumParticles);
        std::iota(tempSortedIndices.begin(), tempSortedIndices.end(), 0);
        std::sort(tempSortedIndices.begin(), tempSortedIndices.end(),
                  [&](size_t A, size_t B) -> bool {
                        return unsortedFiducialData.at(A) < unsortedFiducialData.at(B);
                    });
    }
    std::vector<size_t> const static sortedIndices = tempSortedIndices;

    // Sort the vector
    for (size_t i = 0; i < _fiducialTotalNumParticles; i++)
    {
        fiducialData.at(i) = unsortedFiducialData.at(sortedIndices.at(i));
    }

    // Return the entire dataset fully concatenated and sorted
    return fiducialData;
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