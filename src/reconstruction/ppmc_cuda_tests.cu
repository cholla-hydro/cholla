/*!
 * \file ppmc_cuda_tests.cu
 * \brief Tests for the contents of ppmc_cuda.h and ppmc_cuda.cu
 *
 */

// STL Includes
#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include "../global/global.h"
#include "../io/io.h"
#include "../reconstruction/ppmc_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/hydro_utilities.h"
#include "../utils/testing_utilities.h"

TEST(tHYDROPpmcReconstructor, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 6;
  size_t const ny       = 6;
  size_t const nz       = 6;
  size_t const n_fields = 5;
  double const dx       = doubleRand(prng);
  double const dt       = doubleRand(prng);
  double const gamma    = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(nx * ny * nz * n_fields);
  for (size_t i = 0; i < host_grid.size(); i++) {
    host_grid.at(i) = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.2002498722761787},
                                                                           {734, 1.764334292986655},
                                                                           {950, 3.3600925565746804},
                                                                           {86, 2.4950488327292639}},
                                                                          {{86, 2.4950488327292639},
                                                                           {302, 0.79287723513518138},
                                                                           {518, 1.7614576990062414},
                                                                           {734, 1.8238574169157304},
                                                                           {950, 3.14294317122161}},
                                                                          {{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.0109603398129137},
                                                                           {734, 1.764334292986655},
                                                                           {950, 3.2100231679403066}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{85, 2.6558981128823214},
                                                                            {301, 0.84399195916314151},
                                                                            {517, 1.8381070277226794},
                                                                            {733, 1.764334292986655},
                                                                            {949, 3.0847691079841209}},
                                                                           {{80, 3.1281603739188069},
                                                                            {296, 0.99406757727427164},
                                                                            {512, 1.8732124042412865},
                                                                            {728, 1.6489758692176784},
                                                                            {944, 2.8820015278590443}},
                                                                           {{50, 2.6558981128823214},
                                                                            {266, 0.84399195916314151},
                                                                            {482, 2.0109603398129137},
                                                                            {698, 1.764334292986655},
                                                                            {914, 3.2100231679403066}}};

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size(), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size(), true);

    // Launch kernel
    hipLaunchKernelGGL(PPMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx, ny, nz, dx, dt, gamma, direction);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < host_grid.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.at(direction).find(i) == fiducial_interface_left.at(direction).end())
              ? 0.0
              : fiducial_interface_left.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the right interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (fiducial_interface_right.at(direction).find(i) == fiducial_interface_right.at(direction).end())
                         ? 0.0
                         : fiducial_interface_right.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}

TEST(tMHDPpmcReconstructor, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 6;
  size_t const ny       = 6;
  size_t const nz       = 6;
  size_t const n_fields = 8;
  double const dx       = doubleRand(prng);
  double const dt       = doubleRand(prng);
  double const gamma    = 5.0 / 3.0;

  // Setup host grid. Fill host grid with random values and randomly assign maximum value
  std::vector<double> host_grid(nx * ny * nz * n_fields);
  for (size_t i = 0; i < host_grid.size(); i++) {
    host_grid.at(i) = doubleRand(prng);
  }

  // Allocating and copying to device
  cuda_utilities::DeviceVector<double> dev_grid(host_grid.size());
  dev_grid.cpyHostToDevice(host_grid);

  // Fiducial Data
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.0109603398129137},
                                                                           {734, 1.764334292986655},
                                                                           {950, 6.1966752435374648},
                                                                           {1166, 1.1612148377210372},
                                                                           {1382, 2.4816715896801607}},
                                                                          {{86, 2.2167886449096095},
                                                                           {302, 0.70445164383109971},
                                                                           {518, 2.2081812807712167},
                                                                           {734, 1.9337956878738418},
                                                                           {950, 9.1565812482351436},
                                                                           {1166, 2.8331021062933308},
                                                                           {1382, 1.562787356714062}},
                                                                          {{86, 2.6558981128823214},
                                                                           {302, 0.84399195916314151},
                                                                           {518, 2.0109603398129137},
                                                                           {734, 1.764334292986655},
                                                                           {950, 11.923133284483747},
                                                                           {1166, 1.562787356714062},
                                                                           {1382, 1.1612148377210372}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{85, 2.6558981128823214},
                                                                            {301, 0.84399195916314151},
                                                                            {517, 2.0109603398129137},
                                                                            {733, 1.764334292986655},
                                                                            {949, 8.6490192698558381},
                                                                            {1165, 1.1612148377210372},
                                                                            {1381, 3.1565068702572638}},
                                                                           {{80, 3.3165345946674432},
                                                                            {296, 1.0539291837321079},
                                                                            {512, 1.9599277242665043},
                                                                            {728, 1.8582259623199069},
                                                                            {944, 6.5776143533545097},
                                                                            {1160, 2.8331021062933308},
                                                                            {1376, 1.562787356714062}},
                                                                           {{50, 2.6558981128823214},
                                                                            {266, 0.84399195916314151},
                                                                            {482, 2.0109603398129137},
                                                                            {698, 1.764334292986655},
                                                                            {914, 4.5501389454964283},
                                                                            {1130, 1.562787356714062},
                                                                            {1346, 1.1612148377210372}}};

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(nx * ny * nz * (n_fields - 1), true);
    cuda_utilities::DeviceVector<double> dev_interface_right(nx * ny * nz * (n_fields - 1), true);

    // Launch kernel
    hipLaunchKernelGGL(PPMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx, ny, nz, dx, dt, gamma, direction);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < dev_interface_left.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.at(direction).find(i) == fiducial_interface_left.at(direction).end())
              ? 0.0
              : fiducial_interface_left.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the right interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (fiducial_interface_right.at(direction).find(i) == fiducial_interface_right.at(direction).end())
                         ? 0.0
                         : fiducial_interface_right.at(direction)[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}
