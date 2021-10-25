/*!
 * \file systemTest.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Header file for the tools to run a system test
 * \date 2021-08-30
 *
 */

#pragma once

// STL includes
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// External Libraries and Headers
#include <H5Cpp.h>

/*!
 * \brief This namespace contains one class, SystemTestRunner, whose
 * purpose is to (as you might expect) run system tests.
 */
namespace systemTest
{
    /*!
     * \brief Runs a system test using the full test name to determine all
     * paths.
     *
     * \details By default this class uses the full name of your test, i.e. the test
     * suite name plus the test name, along with some global variables to
     * determine the paths to all the input files. The global variables are all
     * set in main_tests.cpp and are the path to the Cholla directory, the make
     * type being used, and the machine being run on. If the main function does
     * get those it will throw an error so that error checking is not done here.
     *
     * To run a system test simply name the test according to convetion and put
     * the input file in the `cholla/src/system_tests/input_files` directory and
     * the data file in the `cholla/src/system_tests/fiducial_data` directory.
     * Then name the files `testSuiteName_testCaseName` with the `.txt` or `.h5`
     * extension respectively. If this class can't find the files it will
     * throw an error with the path it searched. All the output files from the
     * test are deposited in `cholla/bin/testSuiteName_testCaseName`
     *
     * More advanced functionality is provided with a series of member functions
     * that allow you to programmatically generate the fiducial HDF5 file,
     * choose which datasets to compare, whether or not to compare the number of
     * time steps, etc.
     *
     */
    class SystemTestRunner;
} // namespace systemTest

class systemTest::SystemTestRunner
{
public:
    /// The number of MPI ranks, defaults to 1
    size_t numMpiRanks = 1;

    /*!
     * \brief Run the system test that has been set up
     *
     */
    void runTest();

    /*!
     * \brief Get the Fiducial File object
     *
     * \return H5::H5File
     */
    H5::H5File getFiducialFile(){return _fiducialFile;};

    /*!
     * \brief Get the Test File object
     *
     * \param index The MPI rank of the file you want to return. Defaults to 0
     * \return H5::H5File
     */
    H5::H5File getTestFile(size_t const &i = 0){return _testFileVec[i];};

    /*!
     * \brief Get the vector of datasets that will be tested
     *
     * \return std::vector<std::string>
     */
    std::vector<std::string> getDataSetsToTest(){return _fiducialDataSetNames;};

    /*!
     * \brief Choose which datasets to test. By default it tests all the
     * datasets in the fiducial data. A warning will be thrown if not all the
     * datasets are being tested. Note that any call to this function will
     * overwrite the default values
     *
     * \param[in] dataSetNames A std::vector of std::strings where each entry is
     * a dataset name. Note that it is case sensitive
     */
    void setDataSetsToTest(std::vector<std::string> const &dataSetNames)
        {_fiducialDataSetNames = dataSetNames;};

    /*!
     * \brief Set the Compare Num Time Steps object
     *
     * \param[in] compare Defaults to `true`. If false then the number of timesteps
     * is not compared.
     */
    void setCompareNumTimeSteps(bool const &compare)
    {_compareNumTimeSteps = compare;};

    /*!
     * \brief Set the parameters that Cholla launches with, potentially entirely
     * replacing the need for a settings file
     *
     * \param[in] chollaParams A string of the launch parameters that will
     * override the values in the settings file (if given). Any of Cholla's
     * standard launch paramters work except `outdir` as that is reserved for
     * usage in the systemTest::SystemTestRunner.runTest() method
     */
    void setChollaLaunchParams(std::string const &chollaParams)
    {_chollaLaunchParams = chollaParams;};

    /*!
     * \brief Set or add a fiducial dataset
     *
     * \param[in] fieldName The name of the field to be added
     * \param[in] dataArr The std::shared_ptr for the data array to be added as
     * a data set
     * \param[in] sizeOfArray The number of elements in `dataArr`
     * \param[in] numTimeSteps (optional) Set the `_numFiducialTimeSteps` member variable
     */
    void setFiducialData(std::string const &fieldName,
                         std::shared_ptr<double[]> const &dataArr,
                         size_t const &sizeOfArr,
                         int const &numTimeSteps=-1);

    /*!
     * \brief Generate an array of the specified size populated by the specified
     * value. Return a std::shared_ptr to that array.
     *
     * \param[in] value The value to populate the array with
     * \param[in] nx (optional) The size of the array in the x-direction.
     * Defaults to 1
     * \param[in] ny (optional) The size of the array in the y-direction.
     * Defaults to 1
     * \param[in] nz (optional) The size of the array in the z-direction.
     * Defaults to 1
     * \return std::shared_ptr<double[]> A std::shared_ptr pointing at a
     * 1-dimensional array of the required size
     */
    std::shared_ptr<double[]> generateConstantArray(double const &value,
                                                    size_t const &nx=1,
                                                    size_t const &ny=1,
                                                    size_t const &nz=1);

    /*!
     * \brief Generate an array of the specified size populated by a sine wave.
     * Return a std::shared_ptr to that array. The equation used to generate the
     * wave is:
     *
     * wave = offset + amplitude * sin(kx*xIndex + ky*yIndex + kz*zIndex + phase)
     *
     * \param[in] offset Flat offset from zero
     * \param[in] amplitude Amplitude of the wave
     * \param[in] kx The x component of the wave vector in pixel units
     * \param[in] ky The y component of the wave vector in pixel units
     * \param[in] kz The z component of the wave vector in pixel units
     * \param[in] phase Phase of the sine wave
     * \param[in] nx (optional) The size of the array in the x-direction.
     * Defaults to 1
     * \param[in] ny (optional) The size of the array in the y-direction.
     * Defaults to 1
     * \param[in] nz (optional) The size of the array in the z-direction.
     * Defaults to 1
     * \return std::shared_ptr<double[]>
     */
    std::shared_ptr<double[]> generateSineArray(double const &offset,
                                                double const &amplitude,
                                                double const &kx,
                                                double const &ky,
                                                double const &kz,
                                                double const &phase,
                                                size_t const &nx=1,
                                                size_t const &ny=1,
                                                size_t const &nz=1);

    // Constructor and Destructor
    /*!
     * \brief Construct a new System Test Runner object
     *
     * \param[in] useFiducialFile Indicate if you're using a HDF5 file or will
     * generate your own. Defaults to `true`, i.e. using an HDF5 file. Set to
     * `false` to generate your own
     * \param[in] useSettingsFile Indicate if you're using a settings file. If
     * `true` then the settings file is automatically found based on the naming
     * convention. If false then the user MUST provide all the required settings
     * with the SystemTestRunner::setChollaLaunchParams method
     */
    SystemTestRunner(bool const &useFiducialFile=true,
                     bool const &useSettingsFile=true);
    ~SystemTestRunner();

private:
    /// The fiducial dat file
    H5::H5File _fiducialFile;
    /// The test data file
    std::vector<H5::H5File> _testFileVec;

    /// The path to the Cholla executable
    std::string _chollaPath;
    /// The full name of the test with an underscore instead of a period. This
    /// is the name of many of the input files, the output directory, etc
    std::string _fullTestFileName;
    /// The path to the Cholla settings file
    std::string _chollaSettingsPath;
    /// The path to the fiducial data file
    std::string _fiducialFilePath;
    /// The path to the output directory
    std::string _outputDirectory;
    /// The path and name of the console output file
    std::string _consoleOutputPath;

    /// The user defined parameters to launch Cholla with
    std::string _chollaLaunchParams;

    /// A list of all the data set names in the fiducial data file
    std::vector<std::string> _fiducialDataSetNames;
    /// A list of all the data set names in the test data file
    std::vector<std::string> _testDataSetNames;

    /// The number of fiducial time steps
    int _numFiducialTimeSteps;
    /// Map of fiducial data sets if we're not using a fiducial file
    std::unordered_map<std::string, std::shared_ptr<double[]>> _fiducialDataSets;
    /// Map of the sizes of the fiducial data sets
    std::unordered_map<std::string, size_t> _fiducialArrSize;

    /// Flag to indicate if a fiducial HDF5 data file is being used or a
    /// programmatically generated H5File object. `true` = use a file, `false` =
    /// use generated H5File object
    bool _fiducialFileExists=false;
    /// Flag to choose whether or not to compare the number of time steps
    bool _compareNumTimeSteps=true;

    /*!
    * \brief Move a file. Throws an exception if the file does not exist.
    * or if the move was unsuccessful
    *
    * \param[in] sourcePath The path the the file to be moved
    * \param[in] destinationDirectory The path to the director the file should
    * be moved to
    */
    void _safeMove(std::string const &sourcePath,
                   std::string const &destinationDirectory);

    /*!
    * \brief Checks if the given file exists. Throws an exception if the
    * file does not exist.
    *
    * \param[in] filePath The path to the file to check for
    */
    void _checkFileExists(std::string const &filePath);

    /*!
     * \brief Using GTest assertions to check if the fiducial and test data have
     * the same number of time steps
     *
     */
    void _checkNumTimeSteps();

    /*!
     * \brief Load the test data from the HDF5 file(s). If there is more than
     * one HDF5 file then it concatenates the contents into a single array
     *
     * \param[in] dataSetName The name of the dataset to get
     * \param[out] length The total number of elements in the data set
     * \param[out] testDims An array with the length of each dimension in it
     * \return std::shared_ptr<double[]> A smart pointer pointing to the array
     */
    std::shared_ptr<double[]> _getTestArray(std::string const &dataSetName,
                                            size_t &length,
                                            std::vector<size_t> &testDims);

    /*!
     * \brief Load the fiducial data from an HDF5 file or return user set data
     * array
     *
     * \param[in] dataSetName The name of the dataset to get
     * \param[out] length The total number of elements in the data set
     * \return std::shared_ptr<double[]> A smart pointer pointing to the array
     */
    std::shared_ptr<double[]> _getFiducialArray(std::string const &dataSetName,
                                                size_t &length);

    /*!
     * \brief Return a vector of all the dataset names in the given HDF5 file
     *
     * \param[in] inputFile The HDF5 file to find names in
     * \return std::vector<std::string>
     */
    std::vector<std::string> _findDataSetNames(H5::H5File const &inputFile);
}; // End of class systemTest::SystemTestRunner
