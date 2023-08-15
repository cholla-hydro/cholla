/*!
 * \file systemTest.h
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Header file for the tools to run a system test
 * \date 2021-08-30
 *
 */

#pragma once

// STL includes
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

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
}  // namespace systemTest

class systemTest::SystemTestRunner
{
 public:
  /// The number of MPI ranks, defaults to 1
  size_t numMpiRanks = 1;

  /*!
   * \brief Set the parameters that Cholla launches with, potentially entirely
   * replacing the need for a settings file. A string of the launch parameters
   * that will override the values in the settings file (if given). Any of
   * Cholla's standard launch paramters work except `outdir` as that is
   * reserved for usage in the systemTest::SystemTestRunner.runTest() method
   */
  std::string chollaLaunchParams;

  /*!
   * \brief Run the system test that has been set up
   *
   */
  void runTest(bool const &compute_L2_norm_only = false, double const &maxAllowedL1Error = 0.0,
               double const &maxAllowedError = 0.0);

  /*!
   * \brief Compute the L1 error for each field compared to the initial
   * conditions. Doesn't work with particle data
   *
   * \param[in] maxAllowedL1Error The maximum allowed L1 error for this test
   * \param[in] maxAllowedError The maximum allowed for any value in the test
   *
   */
  void runL1ErrorTest(double const &maxAllowedL1Error, double const &maxAllowedError = 1E-7);

  /*!
   * \brief Launch Cholla as it is set up
   *
   */
  void launchCholla();

  void openHydroTestData();

  /*!
   * \brief Get the Cholla Path object
   *
   * \return std::string The path to the Cholla executable
   */
  std::string getChollaPath() { return _chollaPath; };

  /*!
   * \brief Get the Cholla Settings File Path object
   *
   * \return std::string The full filename/path to the settings file used to
   * initialize Cholla
   */
  std::string getChollaSettingsFilePath() { return _chollaSettingsPath; };

  /*!
   * \brief Get the L2Norm
   *
   * \return double The L2Norm of the last run test
   */
  double getL2Norm() { return _L2Norm; };

  /*!
   * \brief Get the Output Directory object
   *
   * \return std::string The path to the directory where all the output is
   * stored
   */
  std::string getOutputDirectory() { return _outputDirectory; };

  /*!
   * \brief Get the Console Output Path object
   *
   * \return std::string The full filename/path to the file where all the
   * console output is stored
   */
  std::string getConsoleOutputPath() { return _consoleOutputPath; };

  /*!
   * \brief Get the Fiducial File object
   *
   * \return H5::H5File
   */
  H5::H5File getFiducialFile() { return _fiducialFile; };

  /*!
   * \brief Get the Test File object
   *
   * \param index The MPI rank of the file you want to return. Defaults to 0
   * \return H5::H5File
   */
  H5::H5File getTestFile(size_t const &i = 0) { return _testHydroFieldsFileVec[i]; };

  /*!
   * \brief Get the vector of datasets that will be tested
   *
   * \return std::vector<std::string>
   */
  std::vector<std::string> getDataSetsToTest() { return _fiducialDataSetNames; };

  /*!
   * \brief Set the Fixed Epsilon value
   *
   * \param[in] newVal The new value of fixed epsilon
   */
  void setFixedEpsilon(double const &newVal) { _fixedEpsilon = newVal; };

  /*!
   * \brief Choose which datasets to test. By default it tests all the
   * datasets in the fiducial data. A warning will be thrown if not all the
   * datasets are being tested. Note that any call to this function will
   * overwrite the default values
   *
   * \param[in] dataSetNames A std::vector of std::strings where each entry is
   * a dataset name. Note that it is case sensitive
   */
  void setDataSetsToTest(std::vector<std::string> const &dataSetNames) { _fiducialDataSetNames = dataSetNames; };

  /*!
   * \brief Set the Compare Num Time Steps object
   *
   * \param[in] compare Defaults to `true`. If false then the number of
   * timesteps is not compared.
   */
  void setCompareNumTimeSteps(bool const &compare) { _compareNumTimeSteps = compare; };

  /*!
   * \brief Set or add a fiducial dataset
   *
   * \param[in] fieldName The name of the field to be added
   * \param[in] dataArr The std::vector for the data vector to be added as
   * a data set
   */
  void setFiducialData(std::string const &fieldName, std::vector<double> const &dataVec);

  /*!
   * \brief Set the Fiducial Num Time Steps object
   *
   * \param numTimeSteps The number of time steps in the fiducial data
   */
  void setFiducialNumTimeSteps(int const &numTimeSteps) { _numFiducialTimeSteps = numTimeSteps; };

  /*!
   * \brief Generate an vector of the specified size populated by the specified
   * value.
   *
   * \param[in] value The value to populate the vector with
   * \param[in] nx (optional) The size of the field in the x-direction.
   * Defaults to 1
   * \param[in] ny (optional) The size of the field in the y-direction.
   * Defaults to 1
   * \param[in] nz (optional) The size of the field in the z-direction.
   * Defaults to 1
   * \return std::vector<double> A 1-dimensional std::vector of the required
   * size containing the data.
   */
  std::vector<double> generateConstantData(double const &value, size_t const &nx = 1, size_t const &ny = 1,
                                           size_t const &nz = 1);

  /*!
   * \brief Load the test data for physical fields from the HDF5 file(s). If
   * there is more than one HDF5 file then it concatenates the contents into a
   * single vector. Particle data is handeled with _loadTestParticleData
   *
   * \param[in] dataSetName The name of the dataset to get
   * \param[out] testDims An vector with the length of each dimension in it
   * \param[in] file (optional) The vector of HDF5 files to load
   * \return std::vector<double> A vector containing the data
   */
  std::vector<double> loadTestFieldData(std::string dataSetName, std::vector<size_t> &testDims,
                                        std::vector<H5::H5File> file = {});

  /*!
   * \brief Generate a std::vector of the specified size populated by a sine
   * wave. The equation used to generate the wave is:
   *
   * wave = offset + amplitude * sin(kx*xIndex + ky*yIndex + kz*zIndex + phase)
   *
   * \param[in] offset Flat offset from zero
   * \param[in] amplitude Amplitude of the wave
   * \param[in] kx The x component of the wave vector in pixel units
   * \param[in] ky The y component of the wave vector in pixel units
   * \param[in] kz The z component of the wave vector in pixel units
   * \param[in] phase Phase of the sine wave
   * \param[in] nx (optional) The size of the field in the x-direction.
   * Defaults to 1
   * \param[in] ny (optional) The size of the field in the y-direction.
   * Defaults to 1
   * \param[in] nz (optional) The size of the field in the z-direction.
   * Defaults to 1
   * \return std::vector<double> A 1-dimensional std::vector of the required
   * size containing the data.
   */
  std::vector<double> generateSineData(double const &offset, double const &amplitude, double const &kx,
                                       double const &ky, double const &kz, double const &phase, size_t const &nx = 1,
                                       size_t const &ny = 1, size_t const &nz = 1);

  // Constructor and Destructor
  /*!
   * \brief Construct a new System Test Runner object
   *
   * \param[in] particleData Is there particle data?
   * \param[in] hydroData Is there hydro data?
   * \param[in] useFiducialFile Indicate if you're using a HDF5 file or will
   * generate your own. Defaults to `true`, i.e. using an HDF5 file. Set to
   * `false` to generate your own
   * \param[in] useSettingsFile Indicate if you're using a settings file. If
   * `true` then the settings file is automatically found based on the naming
   * convention. If false then the user MUST provide all the required settings
   * with the SystemTestRunner::chollaLaunchParams member variable
   */
  SystemTestRunner(bool const &particleData = false, bool const &hydroData = true, bool const &useFiducialFile = true,
                   bool const &useSettingsFile = true);
  ~SystemTestRunner();

 private:
  /// The fiducial dat file
  H5::H5File _fiducialFile;
  /// The test hydro field data files
  std::vector<H5::H5File> _testHydroFieldsFileVec;
  /// The test particle data files
  std::vector<H5::H5File> _testParticlesFileVec;

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

  /// A list of all the data set names in the fiducial data file
  std::vector<std::string> _fiducialDataSetNames;
  /// A list of all the data set names in the test data file
  std::vector<std::string> _testDataSetNames;

  /// The number of fiducial time steps
  int _numFiducialTimeSteps;
  /// Map of fiducial data sets if we're not using a fiducial file
  std::unordered_map<std::string, std::vector<double>> _fiducialDataSets;

  /// The test particle IDs
  std::vector<double> _testParticleIDs;
  /// The total number of particles in the test dataset
  size_t _testTotalNumParticles = 0;
  /// The fiducial particle IDs
  std::vector<double> _fiducialParticleIDs;
  /// The total number of particles in the fiducial dataset
  size_t _fiducialTotalNumParticles = 0;

  /// Fixed epsilon is changed from the default since AMD/Clang
  /// appear to differ from NVIDIA/GCC/XL by roughly 1E-12
  double _fixedEpsilon = 5.0E-12;

  /// The L2 norm of the error vector
  double _L2Norm;

  /// Flag to indicate if a fiducial HDF5 data file is being used or a
  /// programmatically generated H5File object. `true` = use a file, `false` =
  /// use generated H5File object
  bool _fiducialFileExists = false;
  /// Flag to choose whether or not to compare the number of time steps
  bool _compareNumTimeSteps = true;

  /// Flag to indicate whether or not there is hydro field data
  /// If true then hydro data files are searched for and will be compared to
  /// fiducial values. If false then it is assumed that the test produces no
  /// hydro field data
  bool _hydroDataExists = true;
  /// Flag to indicate whether or not there is particle data
  /// If true then particle data files are searched for and will be compared
  /// to fiducial values. If false then it is assumed that the test produces
  /// no particle data
  bool _particleDataExists = false;

  /*!
   * \brief Using GTest assertions to check if the fiducial and test data have
   * the same number of time steps
   *
   */
  void _checkNumTimeSteps();

  /*!
   * \brief Load the test data for particles from the HDF5 file(s). If
   * there is more than one HDF5 file then it concatenates the contents into a
   * single vector. Field data is handeled with _loadTestFieldData
   *
   * \param[in] dataSetName The name of the dataset to get
   * \return std::vector<double> A vector containing the data
   */
  std::vector<double> _loadTestParticleData(std::string const &dataSetName);

  /*!
   * \brief Load the test data for physical fields from the HDF5 file or
   * returns the user set vector.
   * Particle data is handeled with _loadFiducialParticleData.
   *
   * \param[in] dataSetName The name of the dataset to get
   * \return std::vector<double> A vector with the contents of the data set
   */
  std::vector<double> _loadFiducialFieldData(std::string const &dataSetName);

  /*!
   * \brief Load the fiducial data for particles from the HDF5 file or return
   * the user set vector. Field data is handeled with _loadFiducialFieldData
   *
   * \param[in] dataSetName The name of the dataset to get
   * \return std::vector<double> A vector containing the data
   */
  std::vector<double> _loadFiducialParticleData(std::string const &dataSetName);

  /*!
   * \brief Return a vector of all the dataset names in the given HDF5 file
   *
   * \param[in] inputFile The HDF5 file to find names in
   * \return std::vector<std::string>
   */
  std::vector<std::string> _findDataSetNames(H5::H5File const &inputFile);
};  // End of class systemTest::SystemTestRunner
