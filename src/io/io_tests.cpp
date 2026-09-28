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
  int const num_ranks = 4;

  // Generate the data to read from
  system_test::SystemTestRunner initializer(false, true, false);
  initializer.numMpiRanks = num_ranks;
  initializer.chollaLaunchParams.param("tout", 0.0).param("outstep", 0.0);
  initializer.launchCholla();

  // Reload data and run the test
  int restart_nfile = 0;
  system_test::SystemTestRunner loadRun(false, true, false);
  loadRun.numMpiRanks = num_ranks;
  loadRun.chollaLaunchParams.param("init", "Read_Grid")
      .param("nfile", restart_nfile)
      .param("indir", initializer.getOutputDirectory() + "/" + std::to_string(restart_nfile) + "/");

#ifdef MHD
  loadRun.setFiducialNumTimeSteps(854);
#else   // not MHD
  loadRun.setFiducialNumTimeSteps(427);
#endif  // MHD
  loadRun.runL1ErrorTest(4.2E-7, 5.4E-7);
}
// =============================================================================