
/*!
 * \file cuda_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu), Helena Richie
 * (helenarichie@pitt.edu) \brief Tests for the contents of cuda_utilities.h and
 * cuda_utilities.cpp
 *
 */

// STL Includes
#include <iostream>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../utils/cuda_utilities.h"
#include "../utils/testing_utilities.h"

/*
 PCM : n_ghost = 2
 PLMP : n_ghost = 2
 PLMC : n_ghost = 3
 PPMP : n_ghost = 4
 PPMC : n_ghost = 4
*/

// =============================================================================
// Local helper functions
namespace
{
struct TestParams {
  std::vector<int> n_ghost{2, 2, 3, 4};
  std::vector<int> nx{100, 2048, 2048, 2048};
  std::vector<int> ny{1, 2048, 2048, 2048};
  std::vector<int> nz{1, 4096, 4096, 4096};
  std::vector<std::string> names{"Single-cell 3D PCM/PLMP case", "Large 3D PCM/PLMP case", "Large PLMC case",
                                 "Large PPMP/PPMC case"};
};
}  // namespace

TEST(tHYDROCudaUtilsGetRealIndices, CorrectInputExpectCorrectOutput)
{
  TestParams parameters;
  std::vector<std::vector<int>> fiducial_indices{
      {2, 98, 0, 1, 0, 1}, {2, 2046, 2, 2046, 2, 4094}, {3, 2045, 3, 2045, 3, 4093}, {4, 2044, 4, 2044, 4, 4092}};

  for (size_t i = 0; i < parameters.names.size(); i++) {
    int is;
    int ie;
    int js;
    int je;
    int ks;
    int ke;
    cuda_utilities::Get_Real_Indices(parameters.n_ghost.at(i), parameters.nx.at(i), parameters.ny.at(i),
                                     parameters.nz.at(i), is, ie, js, je, ks, ke);

    std::vector<std::string> index_names{"is", "ie", "js", "je", "ks", "ke"};
    std::vector<int> test_indices{is, ie, js, je, ks, ke};

    for (size_t j = 0; j < test_indices.size(); j++) {
      testing_utilities::checkResults(fiducial_indices[i][j], test_indices[j],
                                      index_names[j] + " " + parameters.names[i]);
    }
  }
}

// =============================================================================
TEST(tALLCompute3DIndices, CorrectInputExpectCorrectOutput)
{
  // Parameters
  int const id = 723;
  int const nx = 34;
  int const ny = 14;

  // Fiducial Data
  int const fiducialXid = 9;
  int const fiducialYid = 7;
  int const fiducialZid = 1;

  // Test Variables
  int testXid;
  int testYid;
  int testZid;

  // Get test data
  cuda_utilities::compute3DIndices(id, nx, ny, testXid, testYid, testZid);

  EXPECT_EQ(fiducialXid, testXid);
  EXPECT_EQ(fiducialYid, testYid);
  EXPECT_EQ(fiducialZid, testZid);
}
// =============================================================================

// =============================================================================
TEST(tALLCompute1DIndex, CorrectInputExpectCorrectOutput)
{
  // Parameters
  int const xid = 72;
  int const yid = 53;
  int const zid = 14;
  int const nx  = 128;
  int const ny  = 64;

  // Fiducial Data
  int const fiducialId = 121544;

  // Test Variable
  int testId;

  // Get test data
  testId = cuda_utilities::compute1DIndex(xid, yid, zid, nx, ny);

  EXPECT_EQ(fiducialId, testId);
}
// =============================================================================
