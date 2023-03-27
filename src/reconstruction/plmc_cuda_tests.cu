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
  size_t const nx       = 4;
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
  std::unordered_map<int, double> fiducial_interface_left  = {{1, 0.76773614979894189},
                                                              {5, 1.927149727335306},
                                                              {9, 2.666157385974266},
                                                              {13, 4.7339225843521469},
                                                              {17, 21.643063389483491}};
  std::unordered_map<int, double> fiducial_interface_right = {{0, 0.76773614979894189},
                                                              {4, 1.927149727335306},
                                                              {8, 2.666157385974266},
                                                              {12, 4.7339225843521469},
                                                              {16, 21.643063389483491}};

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
    hipLaunchKernelGGL(PLMC_cuda, dev_grid.size(), 1, 0, 0, dev_grid.data(), dev_interface_left.data(),
                       dev_interface_right.data(), nx_rot, ny_rot, nz_rot, dx, dt, gamma, direction, n_fields);
    CudaCheckError();
    CHECK(cudaDeviceSynchronize());

    // Perform Comparison
    for (size_t i = 0; i < host_grid.size(); i++) {
      double absolute_diff;
      int64_t ulps_diff;

      // Check the left interface
      double test_val     = dev_interface_left.at(i);
      double fiducial_val = (test_val == 0.0) ? 0.0 : fiducial_interface_left[i];

      EXPECT_TRUE(testingUtilities::nearlyEqualDbl(fiducial_val, test_val, absolute_diff, ulps_diff))
          << "Error in left interface" << std::endl
          << "The fiducial value is:       " << fiducial_val << std::endl
          << "The test value is:           " << test_val << std::endl
          << "The absolute difference is:  " << absolute_diff << std::endl
          << "The ULP difference is:       " << ulps_diff << std::endl;

      // Check the left interface
      test_val     = dev_interface_right.at(i);
      fiducial_val = (test_val == 0.0) ? 0.0 : fiducial_interface_right[i];

      EXPECT_TRUE(testingUtilities::nearlyEqualDbl(fiducial_val, test_val, absolute_diff, ulps_diff))
          << "Error in rigt interface" << std::endl
          << "The fiducial value is:       " << fiducial_val << std::endl
          << "The test value is:           " << test_val << std::endl
          << "The absolute difference is:  " << absolute_diff << std::endl
          << "The ULP difference is:       " << ulps_diff << std::endl;
    }
  }
}
