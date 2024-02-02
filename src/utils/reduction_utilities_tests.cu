/*!
 * \file reduction_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of reduction_utilities.h and
 * reduction_utilities.cpp
 *
 */

// STL Includes
#include <iostream>
#include <random>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../utils/DeviceVector.h"
#include "../utils/cuda_utilities.h"
#include "../utils/reduction_utilities.h"
#include "../utils/testing_utilities.h"

// =============================================================================
// Tests for divergence max reduction
// =============================================================================
TEST(tALLKernelReduceMax, CorrectInputExpectCorrectOutput)
{
  // Launch parameters
  // =================
  cuda_utilities::AutomaticLaunchParams static const launchParams(reduction_utilities::kernelReduceMax);

  // Grid Parameters & testing parameters
  // ====================================
  size_t const gridSize = 64;
  size_t const size     = std::pow(gridSize, 3);
  ;
  Real const maxValue = 4;
  std::vector<Real> host_grid(size);

  // Fill grid with random values and assign maximum value
  std::mt19937 prng(1);
  std::uniform_real_distribution<double> doubleRand(-std::abs(maxValue) - 1, std::abs(maxValue) - 1);
  std::uniform_int_distribution<int> intRand(0, host_grid.size() - 1);
  for (Real& host_data : host_grid) {
    host_data = doubleRand(prng);
  }
  host_grid.at(intRand(prng)) = maxValue;

  // Allocating and copying to device
  // ================================
  cuda_utilities::DeviceVector<Real> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  cuda_utilities::DeviceVector<Real> static dev_max(1);
  dev_max.assign(std::numeric_limits<double>::lowest());

  // Do the reduction
  // ================
  hipLaunchKernelGGL(reduction_utilities::kernelReduceMax, launchParams.numBlocks, launchParams.threadsPerBlock, 0, 0,
                     dev_grid.data(), dev_max.data(), host_grid.size());
  GPU_Error_Check();

  // Perform comparison
  testing_utilities::Check_Results(maxValue, dev_max.at(0), "maximum value found");
}
// =============================================================================
// Tests for divergence max reduction
// =============================================================================
