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
  size_t const nx       = 7;
  size_t const ny       = 7;
  size_t const nz       = 7;
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
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{171, 1.7598055553475744},
                                                                           {514, 3.3921082637175894},
                                                                           {857, 3.5866056366266772},
                                                                           {1200, 3.4794572581328902},
                                                                           {1543, 10.363861270296034}},
                                                                          {{171, 1.6206985712721598},
                                                                           {514, 3.123972986618837},
                                                                           {857, 3.30309596610488},
                                                                           {1200, 3.204417323222251},
                                                                           {1543, 9.544631281899882}},
                                                                          {{171, 1.6206985712721595},
                                                                           {514, 5.0316428671215876},
                                                                           {857, 2.3915465711497186},
                                                                           {1200, 3.2044173232222506},
                                                                           {1543, 12.74302824034023}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{170, 1.7857012385420896},
                                                                            {513, 3.4420234152477129},
                                                                            {856, 3.6393828329638049},
                                                                            {1199, 3.5306577572855762},
                                                                            {1542, 10.516366339570284}},
                                                                           {{164, 1.6206985712721595},
                                                                            {507, 3.1239729866188366},
                                                                            {850, 3.3030959661048795},
                                                                            {1193, 3.2044173232222506},
                                                                            {1536, 9.5446312818998802}},
                                                                           {{122, 1.6206985712721595},
                                                                            {465, 5.4375307473677061},
                                                                            {808, 2.2442413290889327},
                                                                            {1151, 3.2044173232222506},
                                                                            {1494, 13.843305272338561}}};

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
  size_t const nx       = 7;
  size_t const ny       = 7;
  size_t const nz       = 7;
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
  std::vector<std::unordered_map<int, double>> fiducial_interface_left = {{{171, 1.534770576865724},
                                                                           {514, 2.9583427219427034},
                                                                           {857, 3.1279687606328648},
                                                                           {1200, 3.0345219714853804},
                                                                           {1543, 23.015998619464185},
                                                                           {1886, 2.1906071705977261},
                                                                           {2229, 3.1997462690190144}},
                                                                          {{171, 1.6206985712721598},
                                                                           {514, 3.123972986618837},
                                                                           {857, 3.30309596610488},
                                                                           {1200, 3.204417323222251},
                                                                           {1543, 26.732346761532895},
                                                                           {1886, 4.0436839628613175},
                                                                           {2229, 4.1622274705137627}},
                                                                          {{171, 1.6206985712721595},
                                                                           {514, 1.7752459698084133},
                                                                           {857, 3.9720060989313879},
                                                                           {1200, 3.2044173232222506},
                                                                           {1543, 21.984278941312677},
                                                                           {1886, 4.1622274705137627},
                                                                           {2229, 2.1042141607876181}}};

  std::vector<std::unordered_map<int, double>> fiducial_interface_right = {{{170, 1.7925545600850308},
                                                                            {513, 3.4552335159711038},
                                                                            {856, 3.6533503770489086},
                                                                            {1199, 3.5442080266959914},
                                                                            {1542, 29.263332026690119},
                                                                            {1885, 2.1906071705977261},
                                                                            {2228, 3.1997462690190144}},
                                                                           {{164, 1.6206985712721595},
                                                                            {507, 3.1239729866188366},
                                                                            {850, 3.3030959661048795},
                                                                            {1193, 3.2044173232222506},
                                                                            {1536, 26.803126363556764},
                                                                            {1879, 2.1514229421449058},
                                                                            {2222, 4.1622274705137627}},
                                                                           {{122, 1.6206985712721595},
                                                                            {465, 5.4175246353495679},
                                                                            {808, 2.4067132198954435},
                                                                            {1151, 3.2044173232222506},
                                                                            {1494, 35.794674014212731},
                                                                            {1837, 4.1622274705137627},
                                                                            {2180, 2.7068276720054212}}};

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
