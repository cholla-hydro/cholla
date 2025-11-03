/*!
 * \file reduction_utilities_tests.cpp
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Tests for the contents of reduction_utilities.h and
 * reduction_utilities.cpp
 *
 */

// STL Includes
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <limits>
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

long long perform_atomic_min(const std::vector<long long>& host_vals)
{
  // it appears that we need to define the lambda function outside of a googletest
  // test case (which is why this function exists)

  // construct a device vector that holds copies of each host value
  cuda_utilities::DeviceVector<long long> device_vals(host_vals.size());
  device_vals.cpyHostToDevice(host_vals);

  // construct an output buffer where we will write the results
  cuda_utilities::DeviceVector<long long> device_outbuffer(1);
  device_outbuffer.assign(std::numeric_limits<long long>::max());

  // invoke the kernel that we want to check

  const long long* device_vals_ptr = device_vals.data();
  long long* device_out_ptr        = device_outbuffer.data();

  auto loop_fn = [device_vals_ptr, device_out_ptr] __device__(int index) {
    reduction_utilities::backport::atomicMin(device_out_ptr, device_vals_ptr[index]);
  };
  gpuFor(host_vals.size(), loop_fn);
  return device_outbuffer[0];
}

TEST(tALLBackports, AtomicMinLL)
{
  // construct a vector of values to compute the minimum of
  std::vector<long long> host_vals(64);
  for (std::size_t i = 0; i < host_vals.size(); i++) {
    host_vals[i] = static_cast<long long>(i);
  }
  host_vals[1] = host_vals[0];
  host_vals[2] = std::numeric_limits<long long>::min();
  host_vals[3] = std::numeric_limits<long long>::max();

  // get the expected value
  long long expected = *(std::min_element(host_vals.begin(), host_vals.end()));
  long long actual   = perform_atomic_min(host_vals);

  ASSERT_EQ(expected, actual) << "reduction_utilities::backport::atomicMin produced an unexpected result";
}

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
  hipLaunchKernelGGL(reduction_utilities::kernelReduceMax, launchParams.get_numBlocks(),
                     launchParams.get_threadsPerBlock(), 0, 0, dev_grid.data(), dev_max.data(), host_grid.size());
  GPU_Error_Check();

  // Perform comparison
  testing_utilities::Check_Results(maxValue, dev_max.at(0), "maximum value found");
}
// =============================================================================
// Tests for divergence max reduction
// =============================================================================
