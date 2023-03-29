/*!
 * \file plmc_cuda_tests.cu
 * \brief Tests for the contents of plmc_cuda.h and plmc_cuda.cu
 *
 */

// STL Includes
#include <random>
#include <string>
#include <vector>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include <algorithm>

#include "../global/global.h"
#include "../io/io.h"
#include "../reconstruction/plmc_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/testing_utilities.h"

TEST(tHYDROPlmcReconstructor, CorrectInputExpectCorrectOutput)
{
  // Set up PRNG to use
  std::mt19937_64 prng(42);
  std::uniform_real_distribution<double> doubleRand(0.1, 5);

  // Mock up needed information
  size_t const nx       = 5;
  size_t const ny       = 4;
  size_t const nz       = 4;
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
  std::unordered_map<int, double> fiducial_interface_left = {
      {26, 2.1584359129984056},  {27, 0.70033864721549188}, {106, 2.2476363309467553}, {107, 3.0633780053857027},
      {186, 2.2245934101106259}, {187, 2.1015872413794123}, {266, 2.1263341057778309}, {267, 3.9675148506537838},
      {346, 3.3640057502842691}, {347, 21.091316282933843}};
  std::unordered_map<int, double> fiducial_interface_right = {
      {25, 3.8877922383184833},  {26, 0.70033864721549188}, {105, 1.5947787943675635}, {106, 3.0633780053857027},
      {185, 4.0069556576401011}, {186, 2.1015872413794123}, {265, 1.7883678016935785}, {266, 3.9675148506537838},
      {345, 2.8032969746372527}, {346, 21.091316282933843}};

  // Loop over different directions
  for (size_t direction = 0; direction < 1; direction++) {
    // Assign the shape
    size_t nx_rot, ny_rot, nz_rot;
    switch (direction) {
      case 0:
        nx_rot = nx;
        ny_rot = ny;
        nz_rot = nz;
        break;
      case 1:
        nx_rot = ny;
        ny_rot = nz;
        nz_rot = nx;
        break;
      case 2:
        nx_rot = nz;
        ny_rot = nx;
        nz_rot = ny;
        break;
    }

    // Allocate device buffers
    cuda_utilities::DeviceVector<double> dev_interface_left(host_grid.size());
    cuda_utilities::DeviceVector<double> dev_interface_right(host_grid.size());

    // Launch kernel
    hipLaunchKernelGGL(PLMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma, direction, n_fields);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < host_grid.size(); i++) {
      // Check the left interface
      double test_val = dev_interface_left.at(i);
      double fiducial_val =
          (fiducial_interface_left.find(i) == fiducial_interface_left.end()) ? 0.0 : fiducial_interface_left[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "left interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));

      // Check the left interface
      test_val = dev_interface_right.at(i);
      fiducial_val =
          (fiducial_interface_right.find(i) == fiducial_interface_right.end()) ? 0.0 : fiducial_interface_right[i];

      testingUtilities::checkResults(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}
