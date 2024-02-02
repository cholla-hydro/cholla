/*!
 * \file io_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains all the system tests for code in io.h and io.cpp
 *
 */

// External Libraries and Headers
#include <gtest/gtest.h>

// Local includes
#include "../io/io.h"
#include "../system_tests/system_tester.h"

// STL includes
#include <filesystem>
#include <string>

// =============================================================================
TEST(tHYDROtMHDReadGridHdf5, RestartSlowWaveExpectCorrectOutput)
{
  // Set parameters
  int const num_ranks           = 4;
  std::string restart_nfile_str = "0";

  // Generate the data to read from
  system_test::SystemTestRunner initializer(false, true, false);
  initializer.numMpiRanks = num_ranks;
  initializer.chollaLaunchParams.append(" tout=0.0 outstep=0.0");
  initializer.launchCholla();
  std::string const read_directory = initializer.getOutputDirectory() + "/" + restart_nfile_str + "/";

  // Reload data and run the test
  system_test::SystemTestRunner loadRun(false, true, false);
  loadRun.numMpiRanks = num_ranks;
  loadRun.chollaLaunchParams.append(" init=Read_Grid nfile=" + restart_nfile_str + " indir=" + read_directory);

#ifdef MHD
  loadRun.setFiducialNumTimeSteps(854);
#else   // not MHD
  loadRun.setFiducialNumTimeSteps(427);
#endif  // MHD
  loadRun.runL1ErrorTest(4.2E-7, 5.4E-7);
}
// =============================================================================