/*!
 * \file ppmc_cuda_tests.cu
 * \brief Tests for the contents of ppmc_cuda.h and ppmc_cuda.cu
 *
 */

// STL Includes
#include <random>
#include <string>
#include <unordered_map>

// External Includes
#include <gtest/gtest.h>  // Include GoogleTest and related libraries/headers

// Local Includes
#include <algorithm>

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
  size_t const ny       = 1;
  size_t const nz       = 1;
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
  std::unordered_map<int, double> fiducial_interface_left  = {{2, 4.5260179354990537},
                                                              {8, 0.16067557854687248},
                                                              {14, 3.7907707014364083},
                                                              {20, 2.1837489694378442},
                                                              {26, 3.8877922383184833}};
  std::unordered_map<int, double> fiducial_interface_right = {{1, 4.5260179354990537},
                                                              {7, 0.16067557854687248},
                                                              {13, 3.7907707014364083},
                                                              {19, 2.1837489694378442},
                                                              {25, 3.8877922383184833}};

  // Loop over different directions
  for (size_t direction = 0; direction < 3; direction++) {
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
    hipLaunchKernelGGL(PPMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma, direction);
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

      // Check the right interface
      test_val = dev_interface_right.at(i);
      fiducial_val =
          (fiducial_interface_right.find(i) == fiducial_interface_right.end()) ? 0.0 : fiducial_interface_right[i];
      // if (test_val != 0)
      // std::cout << "{" << i << ", " << to_string_exact(test_val) << "}," << std::endl;
      testingUtilities::checkResults(
          fiducial_val, test_val,
          "right interface at i=" + std::to_string(i) + ", in direction " + std::to_string(direction));
    }
  }
}
